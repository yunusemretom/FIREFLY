"""
GNSS noise and jamming signal generator for robustness testing.
"""

import random

class GNSSJammer:
    def __init__(self, jam_level=0.1):
        self.jam_level = jam_level

    def apply_noise(self, coordinates):
        x, y, z = coordinates
        return (
            x + random.uniform(-self.jam_level, self.jam_level),
            y + random.uniform(-self.jam_level, self.jam_level),
            z + random.uniform(-self.jam_level, self.jam_level),
        )
