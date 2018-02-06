"""Tests Numerov equation solver"""
import unittest
import numpy as np
from solver.numerov import numerov_delta


class TestHarmonicOscillator(unittest.TestCase):
    def test_oscillator_5(self):
        """Tests if the harmonic oscillator solution is recovered on the interval [0, 5) with step 0.01
        up to 0.5% accuracy."""
        cos_h = numerov_delta(lambda x: -1, 1, np.cos(0.01), 5, 0.01)
        cos_t = np.cos(np.arange(0, 5, 0.01))

        self.assertTrue(np.allclose(cos_t, cos_h, rtol=0.005, atol=0))

    def test_oscillator_10(self):
        """Tests if the harmonic oscillator solution is recovered on the interval [0, 10) with step 0.01
        up to 0.5% accuracy."""
        cos_h = numerov_delta(lambda x: -1, 1, np.cos(0.01), 10, 0.01)
        cos_t = np.cos(np.arange(0, 10, 0.01))

        self.assertTrue(np.allclose(cos_t, cos_h, rtol=0.005, atol=0))

    def test_oscillator_10_1(self):
        """Tests if the harmonic oscillator solution is recovered on the interval [0, 10) with step 0.02
        up to 0.5% accuracy."""
        cos_h = numerov_delta(lambda x: -1, 1, np.cos(0.02), 10, 0.02)
        cos_t = np.cos(np.arange(0, 10, 0.02))

        self.assertTrue(np.allclose(cos_t, cos_h, rtol=0.005, atol=0))


class TestExponentialFunction(unittest.TestCase):
    def test_exp(self):
        """Tests if the exponential solution is recovered on the interval [0, 5) with step 0.01
        up to 0.5% accuracy."""
        cos_h = numerov_delta(lambda x: 1, 1, np.exp(0.01), 5, 0.01)
        cos_t = np.exp(np.arange(0, 5, 0.01))

        self.assertTrue(np.allclose(cos_t, cos_h, rtol=0.005, atol=0))

    def test_exp_3(self):
        """Tests if the exponential solution is recovered on the interval [0, 0.1) with step 0.01
        up to 2% accuracy. Now the exponential is rising faster and 1% accuracy is not reached."""
        cos_h = numerov_delta(lambda x: 3, 1, np.exp(0.03), 0.1, 0.01)
        cos_t = np.exp(3*np.arange(0, 0.1, 0.01))

        with self.subTest(acc="2%"):
            self.assertTrue(np.allclose(cos_t, cos_h, rtol=0.02, atol=0))
        with self.subTest(acc="1%"):
            self.assertFalse(np.allclose(cos_t, cos_h, rtol=0.01, atol=0))


class TestTooShortArray(unittest.TestCase):
    def test_too_short_array(self):
        """Tests if ValueError is raised for too short arrays"""
        with self.assertRaises(ValueError):
            numerov_delta(lambda x: -1, 1, np.cos(0.02), 10, 20)

if __name__ == '__main__':
    unittest.main()
