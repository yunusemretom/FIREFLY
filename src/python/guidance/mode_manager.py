"""
Flight mode manager for switching between Kalkış, Seyir, and Takip modes.
"""


# mode_manager.py
class ModeManager:
    def __init__(self, config):
        self.state = "KALKIS"
        self.config = config

    def update(self, dist_3d, dz, drone_speed, target_speed, yaw_error, dt):
        safe_target_speed = target_speed if target_speed > 100.0 else 1500.0
        target_speed_cmd = self.config["flight"]["chase_speed"]

        t = self.config["thresholds"]

        if self.state == "KALKIS":
            # Transition once altitude is roughly matched (no speed gate — pitch is 0 during climb)
            if abs(dz) < t["seyir_dz"]:
                self.state = "SEYIR"

        elif self.state == "SEYIR":
            if dist_3d < t["takip_dist"]:
                self.state = "TAKIP"

        elif self.state == "TAKIP":
            # Fly slightly faster than target to close distance, then hold position
            target_speed_cmd = safe_target_speed + self.config["flight"]["takip_speed_margin"]
            if dist_3d > t["seyir_fallback_dist"]:
                self.state = "SEYIR"

        return self.state, target_speed_cmd
