"""Tests Runge-Kutta equation solver"""
import unittest
import numpy as np
from solver import rk


class HarmonicOscillator(unittest.TestCase):
    @staticmethod
    def f(y):
        return np.array([y[1], -y[0]])

    def test_harmonic_motion_initial_amplitude(self):
        """Check harmonic motion. 20 time points"""
        t = np.linspace(0, 2*np.pi, 20)

        solution = rk(self.f, np.array([1, 0]), t)

        with self.subTest(coord="position"):
            self.assertTrue(np.allclose(solution[:, 0], np.cos(t), atol=0.005))

        with self.subTest(coord="velocity"):
            self.assertTrue(np.allclose(solution[:, 1], -np.sin(t), atol=0.005))

    def test_harmonic_motion_initial_velocity(self):
        """Check harmonic motion. 20 time points"""
        t = np.linspace(0, 2*np.pi, 20)

        solution = rk(self.f, np.array([0, 1]), t)

        with self.subTest(coord="position"):
            self.assertTrue(np.allclose(solution[:, 0], np.sin(t), atol=0.005))

        with self.subTest(coord="velocity"):
            self.assertTrue(np.allclose(solution[:, 1], np.cos(t), atol=0.005))

    def test_stable(self):
        """There is no initial amplitude nor velocity"""
        t = np.linspace(0, 2*np.pi, 20)

        solution = rk(self.f, np.array([0, 0]), t)
        self.assertTrue(np.allclose(solution, 0))

if __name__ == '__main__':
    unittest.main()
