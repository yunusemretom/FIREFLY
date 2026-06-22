import unittest
from src.python.navigation.ekf import ExtendedKalmanFilter


class TestEKF(unittest.TestCase):
    def test_initialization(self):
        ekf = ExtendedKalmanFilter()
        self.assertIsNotNone(ekf)


if __name__ == "__main__":
    unittest.main()
