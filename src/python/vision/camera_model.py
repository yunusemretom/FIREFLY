"""
FPV kamera modeli (simulatorde dogrulandi, 2026-11-17).

Sabitler (SDK + olcum): kamera govdeye gore 25 deg yukari bakar, yatay FOV
125 deg, rectilinear (pinhole) projeksiyon. Yakalanan goruntu siyah bantlari
kirpilmis 1600x907 px. Debug truth ile projeksiyon hatasi birkac pikseldir.

Koordinatlar Unreal: X ileri, Y sag, Z yukari (sol el), acilar derece.
Yaw = atan2(vy, vx) bas yonu; dron pitch'i negatif = burun asagi.
"""
import math

import numpy as np


class CameraModel:
    def __init__(self, width=1600, height=907, hfov_deg=125.0, tilt_deg=25.0, roll_sign=1.0):
        self.W = width
        self.H = height
        self.f = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        self.tilt = tilt_deg
        self.roll_sign = roll_sign

    def axes(self, drone_rot):
        """Dunya cercevesinde kamera eksenleri (ileri F, sag R, yukari U)."""
        roll, pitch, yaw = [math.radians(a) for a in drone_rot]
        th = pitch + math.radians(self.tilt)
        F = np.array([math.cos(th) * math.cos(yaw), math.cos(th) * math.sin(yaw), math.sin(th)])
        R0 = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        U0 = np.cross(F, R0)
        if U0[2] < 0:
            U0 = -U0
        # Kamera ekseni etrafinda roll
        phi = self.roll_sign * roll
        R = math.cos(phi) * R0 + math.sin(phi) * U0
        U = -math.sin(phi) * R0 + math.cos(phi) * U0
        return F, R, U

    def project(self, p_world, drone_pos, drone_rot):
        """Dunya noktasi -> (u, v, derinlik). Derinlik <= 0 ise kameranin arkasinda."""
        F, R, U = self.axes(drone_rot)
        d = np.asarray(p_world, float) - np.asarray(drone_pos, float)
        z = float(d @ F)
        if z <= 1e-6:
            return None, None, z
        return self.W / 2 + self.f * float(d @ R) / z, self.H / 2 - self.f * float(d @ U) / z, z

    def ray(self, u, v, drone_rot):
        """Piksel -> dunya cercevesinde birim isin yonu."""
        F, R, U = self.axes(drone_rot)
        r = F + ((u - self.W / 2) / self.f) * R + ((self.H / 2 - v) / self.f) * U
        return r / np.linalg.norm(r)
