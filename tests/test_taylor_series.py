import unittest
import numpy as np
from solver import numerov_taylor_series

np.random.seed(12)


class SinApprox(unittest.TestCase):
    def test_sinus_2(self):
        """Calculates sin(x) up to O(x**4)."""

        for step in [1e-3, 1e-5, 0.1, 0.3]:
            with self.subTest(step=step):
                s = numerov_taylor_series(lambda x: -1, 0, 1, step, 0)
                places = int(4*np.abs(np.log10(step)))
                self.assertAlmostEqual(s, np.sin(step), places=places)

if __name__ == '__main__':
    unittest.main()
