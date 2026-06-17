import tkinter as tk
from tkinter import ttk
import drone_sdk as drone
import threading
import time
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "python", "navigation"))
from nav_integration import NavigationSystem


class DroneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Control Panel - Competition SDK v2.5")
        self.root.geometry("500x850")
        self.root.configure(bg="#0A0A0A")
        
        self.kamikaze_state = "ATTACK" 

        self.nav = NavigationSystem(corrupt_for_test=False, intercept_time=0.0)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#0A0A0A")
        self.style.configure("TLabel", background="#0A0A0A", foreground="#E0E0E0", font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#00FF41") # Matrix Green
        
        self.create_widgets()
        
        self.status_var = tk.StringVar(value="DISCONNECTED")
        self.status_label = tk.Label(root, textvariable=self.status_var, bg="#0A0A0A", fg="#FF3131", font=("Segoe UI", 12, "bold"))
        self.status_label.pack(pady=15)

        self.connect_btn = tk.Button(root, text="INITIALIZE LINK", command=self.toggle_connection, 
                                   bg="#1A1A1A", fg="white", activebackground="#333333", 
                                   activeforeground="#00FF41", relief=tk.FLAT, height=2, width=25,
                                   font=("Segoe UI", 10, "bold"), bd=1, highlightbackground="#00FF41")
        self.connect_btn.pack(pady=5)

        self.running = True
        self.update_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.update_thread.start()

    def create_widgets(self):
        header_frame = tk.Frame(self.root, bg="#0A0A0A")
        header_frame.pack(fill=tk.X, padx=30, pady=(20, 5))
        tk.Label(header_frame, text="SYSTEM TELEMETRY", bg="#0A0A0A", fg="#00FF41", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)

        telemetry_frame = tk.Frame(self.root, bg="#111111", bd=1, relief=tk.SOLID, highlightbackground="#333333")
        telemetry_frame.pack(fill=tk.X, padx=30, pady=5)
        
        self.telemetry_labels = {}
        fields = [
            ("DRONE POS", "0.0, 0.0, 0.0"),
            ("DRONE ROT", "0.0, 0.0, 0.0"),
            ("DRONE SPEED", "0.00 cm/s"),
            ("ALTITUDE", "0.00 cm"),
            ("TARGET POS", "0.0, 0.0, 0.0"),
            ("TARGET SPEED", "0.00 cm/s")
        ]
        
        for i, (label, value) in enumerate(fields):
            tk.Label(telemetry_frame, text=label, bg="#111111", fg="#777777", font=("Segoe UI", 9, "bold")).grid(row=i, column=0, sticky="w", padx=15, pady=4)
            var = tk.StringVar(value=value)
            tk.Label(telemetry_frame, textvariable=var, bg="#111111", fg="#FFFFFF", font=("Consolas", 10)).grid(row=i, column=1, sticky="w", padx=15, pady=4)
            self.telemetry_labels[label] = var

        tk.Label(self.root, text="COMMAND INTERFACE", bg="#0A0A0A", fg="#00FF41", font=("Segoe UI", 14, "bold")).pack(fill=tk.X, padx=30, pady=(25, 10))

        # Autonomous Mode Switch
        self.auto_mode_var = tk.BooleanVar(value=False)
        self.auto_check = tk.Checkbutton(self.root, text="ENABLE KAMIKAZE MODE", variable=self.auto_mode_var, 
                                        bg="#0A0A0A", fg="#00FF41", selectcolor="#000000", 
                                        activebackground="#0A0A0A", activeforeground="#00FF41", 
                                        font=("Segoe UI", 11, "bold"), pady=5)
        self.auto_check.pack()

        max_thr_frame = tk.Frame(self.root, bg="#0A0A0A")
        max_thr_frame.pack(fill=tk.X, padx=30, pady=5)
        tk.Label(max_thr_frame, text="MAX THROTTLE (%):", bg="#0A0A0A", fg="#00FF41", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.max_throttle_slider = tk.Scale(max_thr_frame, from_=0.1, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                           bg="#0A0A0A", fg="#FFFFFF", highlightthickness=0, troughcolor="#1A1A1A",
                                           activebackground="#00FF41", font=("Consolas", 9), command=self.update_thr_slider_range)
        self.max_throttle_slider.set(1.0) # Default 100%
        self.max_throttle_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.sliders_container = tk.Frame(self.root, bg="#0A0A0A")
        self.sliders_container.pack(fill=tk.BOTH, expand=True, padx=30)

        thr_frame = tk.Frame(self.sliders_container, bg="#0A0A0A")
        thr_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        tk.Label(thr_frame, text="THR", bg="#0A0A0A", fg="#00FF41", font=("Segoe UI", 9, "bold")).pack()
        self.throttle_slider = tk.Scale(thr_frame, from_=1.0, to=0.0, resolution=0.01, orient=tk.VERTICAL, length=220, 
                                       bg="#0A0A0A", fg="#FFFFFF", highlightthickness=0, troughcolor="#1A1A1A", 
                                       activebackground="#00FF41", font=("Consolas", 9))
        self.throttle_slider.set(0.0)
        self.throttle_slider.pack()

        axes_frame = tk.Frame(self.sliders_container, bg="#0A0A0A")
        axes_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for axis in ["PITCH", "ROLL", "YAW"]:
            f = tk.Frame(axes_frame, bg="#0A0A0A")
            f.pack(fill=tk.X, pady=8)
            tk.Label(f, text=axis, bg="#0A0A0A", fg="#00FF41", font=("Segoe UI", 9, "bold"), width=7).pack(side=tk.LEFT)
            s = tk.Scale(f, from_=-1.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, 
                        bg="#0A0A0A", fg="#FFFFFF", highlightthickness=0, troughcolor="#1A1A1A", 
                        activebackground="#00FF41", font=("Consolas", 9))
            s.set(0.0)
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            setattr(self, f"{axis.lower()}_slider", s)

        self.arm_var = tk.IntVar(value=0)
        self.arm_check = tk.Checkbutton(self.root, text="ARM WEAPON SYSTEM", variable=self.arm_var, 
                                        bg="#0A0A0A", fg="#FF3131", selectcolor="#000000", 
                                        activebackground="#0A0A0A", activeforeground="#FF3131", 
                                        font=("Segoe UI", 12, "bold"), pady=20)
        self.arm_check.pack()

    def update_thr_slider_range(self, val):
        new_max = float(val)
        current_val = self.throttle_slider.get()
        self.throttle_slider.configure(from_=new_max)
        if current_val > new_max:
            self.throttle_slider.set(new_max)

    def toggle_connection(self):
        if not drone.is_connected():
            if drone.connect():
                self.status_var.set("LINK ESTABLISHED")
                self.status_label.config(fg="#00FF41")
                self.connect_btn.config(text="TERMINATE LINK", bg="#330000")
        else:
            drone.disconnect()
            self.status_var.set("DISCONNECTED")
            self.status_label.config(fg="#FF3131")
            self.connect_btn.config(text="INITIALIZE LINK", bg="#1A1A1A")

    def control_loop(self):
      
        KP_YAW = 0.01  #yawda sallantı var
        KP_PITCH = 0.035
        KP_THR = 0.01
        
        BASE_HOVER_THR = 0.58 

        last_tar_speed = 0.0

        while self.running:
            if drone.is_connected():
                if self.auto_mode_var.get():
                    t = drone.get_telemetry()
                    d_rot = t["drone"]["rotation"]
                    d_speed = t["drone"]["speed"]

                    nav_out = self.nav.process(t)
                    dx = nav_out["dx"]
                    dy = nav_out["dy"]
                    dz = nav_out["dz"]
                    dist_3d = nav_out["dist_3d"]
                    dist_2d = nav_out["dist_2d"]
                    tar_speed = nav_out["target_speed"]   
                    last_tar_speed = tar_speed
                    
                    self.status_var.set("TARGET LOCKED - ATTACKING")
                    self.status_label.config(fg="#00FF41")

                    # 60 metreden itibaren strateji başlamakta
                    proximity_scale = min(1.0, dist_3d / 6000.0)
                    
                    # YAW: Yatayda hedefe kilitlenme kısmı
                    target_yaw = math.degrees(math.atan2(dy, dx))
                    yaw_error = target_yaw - d_rot[2]
                    while yaw_error > 180: yaw_error -= 360
                    while yaw_error < -180: yaw_error += 360
                    final_yaw = yaw_error * KP_YAW
                    
                    CAMERA_TILT = 25.0
                    target_pitch_world = math.degrees(math.atan2(dz, dist_2d))
                    lock_pitch = target_pitch_world - CAMERA_TILT
                    
                    # Faz geçiş mesafesi: 45 metre verildi
                    PHASE2_START = 4500.0
                    max_allowed_thr = self.max_throttle_slider.get()
                    
                    #  (-25 derece) verdim ki hedefe yetişebilsin
                    DIVE_OFFSET = -25.0 

                    # Hedefi yakalamak için gereken tahmini gaz (Match Speed)
                    match_target_thr = (BASE_HOVER_THR * 0.92) + (tar_speed / 3800.0)
                    
                    alt_correction = dz * 0.0015 
                    match_target_thr += alt_correction

                    if dist_3d > PHASE2_START:
                        target_pitch_drone = lock_pitch + DIVE_OFFSET
                        
                        final_thr = min(match_target_thr + 0.24, max_allowed_thr)
                    else:
                        local_scale = max(0.0, dist_3d / PHASE2_START)
                        
                        flare_trigger_scale = min(1.0, dist_3d / 1000.0)
                        flare_effect = (1.0 - flare_trigger_scale) * 15.0 
                        
                        target_pitch_drone = lock_pitch + (DIVE_OFFSET * local_scale) + flare_effect
                        
                        approach_buffer = 0.22 * local_scale
                        final_thr = match_target_thr + approach_buffer

                    speed_gap = max(0.0, tar_speed - d_speed)
                    final_thr += min(0.42, speed_gap / 3200.0)

                    final_thr = min(max_allowed_thr, final_thr)
                    
                    pitch_error = target_pitch_drone - d_rot[1]
                    final_pitch = pitch_error * KP_PITCH
                    
                    final_roll = -d_rot[0] * 0.5
                    
                    fy = max(-1.0, min(1.0, final_yaw))
                    fp = max(-1.0, min(1.0, final_pitch))
                    fr = max(-1.0, min(1.0, final_roll))
                    drone.set_yaw(fy)
                    drone.set_pitch(fp)
                    drone.set_throttle(max(-0.8, min(self.max_throttle_slider.get(), final_thr)))
                    drone.set_roll(fr)
                    drone.set_arm(True)
                    
                    self.yaw_slider.set(fy)
                    self.pitch_slider.set(fp)
                    self.throttle_slider.set(max(-0.8, min(self.max_throttle_slider.get(), final_thr)))
                    self.arm_var.set(1)
                else:
                    self.kamikaze_state = "READY"
                    if drone.is_connected():
                        self.status_var.set("LINK ESTABLISHED")
                        self.status_label.config(fg="#00FF41")
                    
                    drone.set_control_surfaces(
                        self.throttle_slider.get(),
                        self.pitch_slider.get(),
                        self.roll_slider.get(),
                        self.yaw_slider.get(),
                        bool(self.arm_var.get()),
                    )
                
                t = drone.get_telemetry()
                d = t["drone"]
                tar = t["target"]
                
                self.telemetry_labels["DRONE POS"].set(f"{d['position'][0]:.1f}, {d['position'][1]:.1f}, {d['position'][2]:.1f}")
                self.telemetry_labels["DRONE ROT"].set(f"{d['rotation'][0]:.1f}, {d['rotation'][1]:.1f}, {d['rotation'][2]:.1f}")
                self.telemetry_labels["DRONE SPEED"].set(f"{d['speed']:.2f} cm/s")
                self.telemetry_labels["ALTITUDE"].set(f"{d['altitude']:.2f} cm")
                self.telemetry_labels["TARGET POS"].set(f"{tar['position'][0]:.1f}, {tar['position'][1]:.1f}, {tar['position'][2]:.1f}")
                if self.auto_mode_var.get():
                    self.telemetry_labels["TARGET SPEED"].set(f"{last_tar_speed:.2f} cm/s (EKF)")
                else:
                    self.telemetry_labels["TARGET SPEED"].set(f"{tar.get('speed', 0.0):.2f} cm/s")
            
            time.sleep(0.05)

if __name__ == "__main__":
    root = tk.Tk()
    gui = DroneGUI(root)
    root.mainloop()
    drone.disconnect()