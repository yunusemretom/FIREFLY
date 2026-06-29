
import numpy as np
from collections import deque


class GNSSCorruptor:

    def __init__(self,
                 noise_std=15.0,         
                 jump_prob=0.02,        
                 jump_magnitude=5000.0, 
                 dropout_prob=0.10,     
                 offset=(0.0, 0.0, 0.0), 
                 delay_steps=0,         
                 enabled=True,
                 seed=None):
        self.noise_std = noise_std
        self.jump_prob = jump_prob
        self.jump_magnitude = jump_magnitude
        self.dropout_prob = dropout_prob
        self.offset = np.array(offset, dtype=float)
        self.delay_steps = delay_steps
        self.enabled = enabled
        self.rng = np.random.default_rng(seed)
        self._delay_buf = deque(maxlen=max(1, delay_steps + 1))

    def corrupt(self, pos):
        if not self.enabled or pos is None:
            return pos

        if self.rng.random() < self.dropout_prob:
            return None

        p = np.array(pos, dtype=float) + self.offset
        p += self.rng.normal(0.0, self.noise_std, size=3)

        if self.rng.random() < self.jump_prob:
            p += self.rng.normal(0.0, self.jump_magnitude, size=3)

        if self.delay_steps > 0:
            self._delay_buf.append(tuple(p))
            if len(self._delay_buf) < self._delay_buf.maxlen:
                return tuple(p)
            return self._delay_buf[0]

        return tuple(p)


class GNSSPrefilter:
    def __init__(self, max_abs=1.0e7, origin_eps=1.0):
        self.max_abs = max_abs
        self.origin_eps = origin_eps

    def prefilter(self, pos):
        if pos is None:
            return None
        if any((v is None or np.isnan(v) or np.isinf(v)) for v in pos):
            return None
        if any(abs(v) > self.max_abs for v in pos):
            return None
        # (0,0,0) veya orijine cok yakin -> baglanti baslangici cop frame'i, reddet
        if all(abs(v) < self.origin_eps for v in pos):
            return None
        return pos