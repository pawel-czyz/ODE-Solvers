"""Test derivative function"""
import unittest
from solver.derivatives import derivative


class TestSquare(unittest.TestCase):
    """Test derivatives with square function x -> x**2"""
    @staticmethod
    def f(x, order=0):
        if order == 0:
            return x**2
        elif order == 1:
            return 2*x
        elif order == 2:
            return 2

    def test(self):
        """Test values at many points"""
        for point in [0, 1, -1, -2.35, -20, -0.501, -8, 12]:
            for order in range(3):
                with self.subTest(point=point, order=order):
                    self.assertAlmostEqual(self.f(point, order), derivative(self.f, point, order), places=7)


class TestPolynomial(unittest.TestCase):
    """Test derivatives with a polynomial of 4th degree"""
    @staticmethod
    def f(x, order=0):
        if order == 0:
            return x**4 + 2*x**2 + 0.1
        elif order == 1:
            return 4*x**3 + 4*x
        elif order == 2:
            return 12*x**2 + 4

    def test(self):
        """Test values at many points"""
        for point in [0, 1, -1, -2.35, -20, -0.501, -8, 12]:
            for order in range(3):
                with self.subTest(point=point, order=order):
                    self.assertAlmostEqual(self.f(point, order), derivative(self.f, point, order), places=3)


class TestExceptions(unittest.TestCase):
    """Test if appropriate exceptions are raised"""
    def test_wrong_order(self):
        for order in [-10, -5, -2, -1, 3, 4, 8]:
            with self.assertRaises(ValueError):
                derivative(lambda x: x, 0, order)

if __name__ == '__main__':
    unittest.main()
