"""
GNSS filter for smoothing and rejecting outliers from corrupted GNSS data.(burda bozuk GPS simülasyonu + ölçüm ön-eleme)
"""


import numpy as np


class GNSSCorruptor:

    def __init__(self,
                 noise_std=15.0,      # Gaussian gürültü std cm cinsinden girildi, deişebilir
                 jump_prob=0.02,      # her ölçümde sıçrama ihtimali
                 jump_magnitude=5000.0,  # sıçrama büyüklüğü yine cm cinsinden dğeilişebilir
                 dropout_prob=0.10,   # ölçümün tamamen kaybolma ihtimali
                 enabled=True,
                 seed=None):
        self.noise_std = noise_std
        self.jump_prob = jump_prob
        self.jump_magnitude = jump_magnitude
        self.dropout_prob = dropout_prob
        self.enabled = enabled
        self.rng = np.random.default_rng(seed)

    def corrupt(self, pos):

        if not self.enabled or pos is None:
            return pos

        if self.rng.random() < self.dropout_prob:
            return None

        px, py, pz = pos

        px += self.rng.normal(0, self.noise_std)
        py += self.rng.normal(0, self.noise_std)
        pz += self.rng.normal(0, self.noise_std)

        if self.rng.random() < self.jump_prob:
            px += self.rng.normal(0, self.jump_magnitude)
            py += self.rng.normal(0, self.jump_magnitude)
            pz += self.rng.normal(0, self.jump_magnitude)

        return (px, py, pz)


class GNSSPrefilter:

    def __init__(self, max_abs=1.0e7):
        self.max_abs = max_abs  

    def prefilter(self, pos):
        if pos is None:
            return None
        if any((v is None or np.isnan(v) or np.isinf(v)) for v in pos):
            return None
        if any(abs(v) > self.max_abs for v in pos):
            return None
        return pos