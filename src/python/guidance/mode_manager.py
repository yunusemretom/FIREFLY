class ModeManager:
    def __init__(self, config):
        self.state = "KALKIS"
        self.lock_on_timer = 0.0
        self.config = config

    def update(self, dist_3d, dz, drone_speed, target_speed, yaw_error, dt):
        safe_target_speed = target_speed if target_speed > 100.0 else 1500.0
        target_speed_cmd = self.config['flight']['chase_speed']
        
        t = self.config['thresholds']

        if self.state == "KALKIS":
            if abs(dz) < t['seyir_dz'] and drone_speed > 500:
                self.state = "SEYIR"
                
        elif self.state == "SEYIR":
            if dist_3d < t['takip_dist']:
                self.state = "TAKIP"
                self.lock_on_timer = 0.0
                
        elif self.state == "TAKIP":
            target_speed_cmd = safe_target_speed + self.config['flight']['takip_speed_margin']
            lock_ok = (abs(yaw_error) < t['lock_yaw_max'] and 
                       abs(dz) < t['lock_dz_max'] and 
                       dist_3d < t['lock_dist_max'])
            
            if lock_ok:
                self.lock_on_timer += dt
            else:
                self.lock_on_timer = max(0.0, self.lock_on_timer - dt * 0.5)

            if self.lock_on_timer >= t['lock_on_secs']:
                self.state = "ANGAJMAN"
            
            if dist_3d > t['seyir_fallback_dist']:
                self.state = "SEYIR"
                self.lock_on_timer = 0.0

        elif self.state == "ANGAJMAN":
            target_speed_cmd = self.config['flight']['chase_speed'] + self.config['flight']['angajman_speed_boost']
            if dist_3d > t['angajman_miss_dist']:
                self.state = "SEYIR"
                self.lock_on_timer = 0.0

        return self.state, target_speed_cmd
