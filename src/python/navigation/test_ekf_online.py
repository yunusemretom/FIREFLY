
import sys
sys.path.insert(0, "/tmp/out/navigation")
import numpy as np
from ekf import EKF
from gnss_filter import GNSSCorruptor, GNSSPrefilter
from waypoint_controller import WaypointController

def run(noise, jump_p, jump_mag, drop_p, sigma_a, chi2, R, label):
    np.random.seed(0); dt=0.05; T=600
    pos=np.array([0.,0.,5000.]); vel=np.array([1500.,200.,0.])
    truth=[]
    for k in range(T):
        if 200<k<300: vel[1]+=80.0*dt*10
        pos=pos+vel*dt; truth.append(pos.copy())
    truth=np.array(truth)
    cor=GNSSCorruptor(noise_std=noise,jump_prob=jump_p,jump_magnitude=jump_mag,dropout_prob=drop_p,enabled=True,seed=1)
    pre=GNSSPrefilter(); ekf=EKF(sigma_a=sigma_a,chi2_threshold=chi2,r_pos=R)
    er,ee=[],[]; nd=nr=0
    for k in range(T):
        z=pre.prefilter(cor.corrupt(tuple(truth[k])))
        if z is None: nd+=1
        est,acc=ekf.step(z,dt=dt)
        if z is not None and not acc: nr+=1
        if z is not None: er.append(np.linalg.norm(np.array(z)-truth[k]))
        ee.append(np.linalg.norm(np.array(est)-truth[k]))
    er=np.array(er); ee=np.array(ee[60:])
    print(f"[{label}] drop={nd} reject={nr} | ham RMSE={np.sqrt((er**2).mean()):7.0f} "
          f"EKF RMSE={np.sqrt((ee**2).mean()):7.0f} max={ee.max():7.0f} "
          f"-> {np.sqrt((er**2).mean())/max(1,np.sqrt((ee**2).mean())):.1f}x")

run(50, 0.01, 3000, 0.08, sigma_a=900, chi2=11.34, R=2500, label="gercekci, chi2=11.34")
run(50, 0.01, 3000, 0.08, sigma_a=900, chi2=16.27, R=2500, label="gercekci, chi2=16.27")
run(50, 0.01, 3000, 0.08, sigma_a=1500,chi2=16.27, R=2500, label="gercekci, sigma_a=1500")
run(20, 0.03, 8000, 0.15, sigma_a=1500,chi2=16.27, R=2500, label="zor")