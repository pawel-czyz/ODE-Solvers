import unittest
from solver import gradient_descent


class TestSquare(unittest.TestCase):
    @staticmethod
    def loss(x):
        return x*x

    def test_basic(self):
        """Test if a minimum of a square function is found."""
        xm, vm = gradient_descent(self.loss, 0.2)

        with self.subTest("Point"):
            self.assertLess(abs(xm), 0.05)
        with self.subTest("Value"):
            self.assertEqual(vm, self.loss(xm))

    def test_n_steps_influence(self):
        """Tests whether bigger step number results in better accuracy."""
        x10, _ = gradient_descent(self.loss, 0.2, n_steps=10)
        x20, _ = gradient_descent(self.loss, 0.2, n_steps=20)
        self.assertLess(abs(x20), abs(x10))

    def test_far_away(self):
        """Start from value lying very far from the global maximum."""
        xm, _ = gradient_descent(self.loss, 200, n_steps=40)

        with self.subTest("Point"):
            self.assertLess(abs(xm), 0.05)


class TestTwoMinimas(unittest.TestCase):
    @staticmethod
    def loss(x):
        return x**2 * (x-10)**2

    def test_left_minimum(self):
        x, _ = gradient_descent(self.loss, 2, step=0.001, n_steps=50)
        self.assertLess(abs(x), 0.05)

    def test_right_minimum(self):
        x, _ = gradient_descent(self.loss, 8, step=0.001, n_steps=40)
        self.assertLess(abs(x-10), 0.05)

if __name__ == '__main__':
    unittest.main()
