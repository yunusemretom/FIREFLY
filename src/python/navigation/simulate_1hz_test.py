"""
1Hz Kosulu Simulasyonu — Yarisma Sartlarinda Beklenen Performans Testi
==========================================================================
Gercek yuksek frekansli (20Hz) yorunge, tam 1Hz araliklarla ornekleniyor
(rakip ~4s araliklar yerine gercek yarisma kosulu). Olculmus gurultu
istatistikleri (std_x=1426.5, std_y=1574.9, std_z=385.1) ile sentetik
GPS gurultusu ekleniyor. Ardindan CTRV UKF ve CV Kalman karsilastirmali
test ediliyor.
"""

import numpy as np
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

np.random.seed(7)

df = pd.read_excel('/home/tom/Documents/Projeler/Firefly/hedef_konumlari.xlsx')
t_hi = df['Time_Relative'].values
rx_hi, ry_hi, rz_hi = df['Target_Real_X'].values, df['Target_Real_Y'].values, df['Target_Real_Z'].values

# ---------------------------------------------------------------------
# 1Hz ornekleme (gercek yarisma kosulu)
# ---------------------------------------------------------------------
t1hz = np.arange(np.ceil(t_hi.min()), np.floor(t_hi.max()), 1.0)
real_x = np.interp(t1hz, t_hi, rx_hi)
real_y = np.interp(t1hz, t_hi, ry_hi)
real_z = np.interp(t1hz, t_hi, rz_hi)

R_STD = np.array([1426.5, 1574.9, 385.1])
noisy_x = real_x + np.random.normal(0, R_STD[0], len(t1hz))
noisy_y = real_y + np.random.normal(0, R_STD[1], len(t1hz))
noisy_z = real_z + np.random.normal(0, R_STD[2], len(t1hz))


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


raw_err = np.sqrt((noisy_x - real_x)**2 + (noisy_y - real_y)**2 + (noisy_z - real_z)**2)
print(f"HAM (sentetik 1Hz gurultulu) ortalama hata: {raw_err.mean():.1f}")


# =======================================================================
# 1) 3D CV Kalman
# =======================================================================
class CV3DKalman:
    def __init__(self, q_std, r_std):
        self.q_std = q_std
        self.R = np.diag(r_std ** 2)

    def init(self, px, py, pz):
        self.x = np.array([px, py, pz, 0.0, 0.0, 0.0])
        self.P = np.diag(list(self.R.diagonal()) + [400.0**2]*3)

    def predict(self, dt):
        F = np.eye(6); F[0,3]=F[1,4]=F[2,5]=dt
        q = self.q_std**2
        dt2,dt3,dt4 = dt**2,dt**3,dt**4
        Q = np.zeros((6,6))
        for i in range(3):
            Q[i,i]=0.25*dt4*q; Q[i,i+3]=0.5*dt3*q; Q[i+3,i]=0.5*dt3*q; Q[i+3,i+3]=dt2*q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.zeros((3,6)); H[0,0]=H[1,1]=H[2,2]=1
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P


def run_cv(q_std):
    kf = CV3DKalman(q_std, R_STD)
    kf.init(noisy_x[0], noisy_y[0], noisy_z[0])
    out = [kf.x.copy()]
    for i in range(1, len(t1hz)):
        dt = t1hz[i] - t1hz[i-1]
        kf.predict(dt)
        kf.update(np.array([noisy_x[i], noisy_y[i], noisy_z[i]]))
        out.append(kf.x.copy())
    return np.array(out)


best_cv = None
for q_std in [10,25,50,75,100,150,200,300,450,650,900]:
    filt = run_cv(q_std)
    err = rmse(filt[:,0],real_x)+rmse(filt[:,1],real_y)+rmse(filt[:,2],real_z)
    if best_cv is None or err < best_cv[0]:
        best_cv = (err, q_std)
filt_cv = run_cv(best_cv[1])
cv_err = np.sqrt((filt_cv[:,0]-real_x)**2+(filt_cv[:,1]-real_y)**2+(filt_cv[:,2]-real_z)**2)
print(f"CV KALMAN (q_std={best_cv[1]}) ortalama hata: {cv_err.mean():.1f}  |  iyilesme: {(1-cv_err.mean()/raw_err.mean())*100:.1f}%")


# =======================================================================
# 2) CTRV UKF
# =======================================================================
SCALE = 1000.0

def fx(x, dt):
    px,py,yaw,v,yaw_rate,pz,vz = x
    if abs(yaw_rate) > 1e-4:
        px_n = px + (v/yaw_rate)*(np.sin(yaw+yaw_rate*dt)-np.sin(yaw))
        py_n = py + (v/yaw_rate)*(-np.cos(yaw+yaw_rate*dt)+np.cos(yaw))
    else:
        px_n = px + v*np.cos(yaw)*dt
        py_n = py + v*np.sin(yaw)*dt
    yaw_n = yaw + yaw_rate*dt
    pz_n = pz + vz*dt
    return np.array([px_n,py_n,yaw_n,v,yaw_rate,pz_n,vz])

def hx(x):
    return np.array([x[0],x[1],x[5]])

def state_mean(sigmas, Wm):
    x = np.zeros(7)
    for idx in [0,1,3,4,5,6]:
        x[idx] = np.sum(sigmas[:,idx]*Wm)
    x[2] = np.arctan2(np.sum(np.sin(sigmas[:,2])*Wm), np.sum(np.cos(sigmas[:,2])*Wm))
    return x

def state_residual(a,b):
    y = a-b
    y[2] = (y[2]+np.pi)%(2*np.pi)-np.pi
    return y

def process_noise(dt, std_a, std_yawdd, std_vz):
    Q = np.zeros((7,7)); dt2=dt**2
    Q[0,0]=Q[1,1]=0.25*dt2*dt2*std_a**2
    Q[3,3]=dt2*std_a**2
    Q[2,2]=0.25*dt2*dt2*std_yawdd**2
    Q[4,4]=dt2*std_yawdd**2
    Q[5,5]=0.25*dt2*dt2*std_vz**2
    Q[6,6]=dt2*std_vz**2
    return Q

def build_ukf(r_pos_std):
    points = MerweScaledSigmaPoints(n=7, alpha=1.0, beta=2.0, kappa=0.0, subtract=state_residual)
    ukf = UKF(dim_x=7, dim_z=3, fx=fx, hx=hx, dt=1.0, points=points, x_mean_fn=state_mean, residual_x=state_residual)
    ukf.R = np.diag(np.array(r_pos_std)**2)
    return ukf

def run_ctrv(std_a, std_yawdd, std_vz):
    r_pos_std_scaled = [s/SCALE for s in R_STD]
    ukf = build_ukf(r_pos_std_scaled)
    dx0=(noisy_x[1]-noisy_x[0])/SCALE; dy0=(noisy_y[1]-noisy_y[0])/SCALE
    dt0=t1hz[1]-t1hz[0]
    yaw0=np.arctan2(dy0,dx0); v0=np.hypot(dx0,dy0)/dt0
    ukf.x = np.array([noisy_x[0]/SCALE, noisy_y[0]/SCALE, yaw0, v0, 0.0, noisy_z[0]/SCALE, 0.0])
    ukf.P = np.diag([r_pos_std_scaled[0]**2, r_pos_std_scaled[1]**2, 0.5, (v0*0.3)**2+1.0, 0.1, r_pos_std_scaled[2]**2, (50/SCALE)**2])
    out = [ukf.x.copy()]
    for i in range(1, len(t1hz)):
        dt = t1hz[i]-t1hz[i-1]
        ukf.Q = process_noise(dt, std_a/SCALE, std_yawdd, std_vz/SCALE)
        ukf.predict(dt=dt)
        ukf.P = 0.5*(ukf.P+ukf.P.T) + np.eye(7)*1e-6
        z = np.array([noisy_x[i]/SCALE, noisy_y[i]/SCALE, noisy_z[i]/SCALE])
        ukf.update(z)
        ukf.P = 0.5*(ukf.P+ukf.P.T) + np.eye(7)*1e-6
        out.append(ukf.x.copy())
    out = np.array(out)
    out[:, [0,1,3,5,6]] *= SCALE
    return out

best_ctrv = None
for std_a in [20,50,100,150,250,400]:
    for std_yawdd in [0.02,0.05,0.1,0.15,0.25,0.4]:
        filt = run_ctrv(std_a, std_yawdd, 50.0)
        err = rmse(filt[:,0],real_x)+rmse(filt[:,1],real_y)+rmse(filt[:,5],real_z)
        if best_ctrv is None or err < best_ctrv[0]:
            best_ctrv = (err, std_a, std_yawdd)

filt_ctrv = run_ctrv(best_ctrv[1], best_ctrv[2], 50.0)
ctrv_err = np.sqrt((filt_ctrv[:,0]-real_x)**2+(filt_ctrv[:,1]-real_y)**2+(filt_ctrv[:,5]-real_z)**2)
print(f"CTRV UKF (std_a={best_ctrv[1]}, std_yawdd={best_ctrv[2]}) ortalama hata: {ctrv_err.mean():.1f}  |  iyilesme: {(1-ctrv_err.mean()/raw_err.mean())*100:.1f}%")

# ---------------------------------------------------------------------
np.save('/tmp/sim_1hz_results.npy', {
    't': t1hz, 'real_x': real_x, 'real_y': real_y, 'real_z': real_z,
    'noisy_x': noisy_x, 'noisy_y': noisy_y, 'noisy_z': noisy_z,
    'cv': filt_cv, 'ctrv': filt_ctrv,
    'raw_err': raw_err, 'cv_err': cv_err, 'ctrv_err': ctrv_err,
    'best_cv_q': best_cv[1], 'best_ctrv': (best_ctrv[1], best_ctrv[2]),
})
