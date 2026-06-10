"""
Flight mode manager for switching between Manuel, Autonomous (Kamikaze), and FailSafe modes.
"""

class ModeManager:
    def __init__(self):
        self.current_mode = "MANUAL"

    def set_mode(self, mode):
        self.current_mode = mode
