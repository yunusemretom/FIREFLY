
class ModeManager:
    def __init__(self, config):
        self.state = "KALKIS"
        self.config = config

    def update(self, dist_3d, dz, drone_speed, target_speed, yaw_error, dt):
        safe_target_speed = target_speed if target_speed > 100.0 else 1500.0
        target_speed_cmd = self.config["flight"]["chase_speed"]

        t = self.config["thresholds"]

        angajman_dist = t.get("angajman_dist", 4500.0)

        if self.state == "KALKIS":
            if abs(dz) < t["seyir_dz"]:
                self.state = "SEYIR"

        elif self.state == "SEYIR":
            if dist_3d < t["takip_dist"]:
                self.state = "TAKIP"

        elif self.state == "TAKIP":
            target_speed_cmd = safe_target_speed + self.config["flight"]["takip_speed_margin"]
            if dist_3d > t["seyir_fallback_dist"]:
                self.state = "SEYIR"
  
            elif 0.0 < dist_3d < angajman_dist:
                self.state = "ANGAJMAN"

        elif self.state == "ANGAJMAN":

            engagement_margin = self.config["flight"].get(
                "angajman_speed_margin",
                self.config["flight"]["takip_speed_margin"] * 2.0,
            )
            target_speed_cmd = safe_target_speed + engagement_margin
            if dist_3d > angajman_dist * 1.3:
                self.state = "TAKIP"

        return self.state, target_speed_cmd