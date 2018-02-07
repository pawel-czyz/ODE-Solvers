import sys
sys.path.append(".")
from solver.numerov import numerov_delta
from solver.derivatives import numerov_taylor_series, derivative
import numpy as np
import matplotlib.pyplot as plt


def hermite_polynomial(n):
    """Returns Hermite polynomial of order `n`

    Parameters
    ----------
    n : int
        non-negative integer

    Returns
    -------
    func
        Hermite polynomial `f` of order `n`

    Raises
    ------
    TypeError
        if `n` is not int
    ValueError
        if `n` is negative
    """

    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")

    if n == 0:
        return lambda x: 1
    elif n == 1:
        return lambda x: 2*x
    else:
        return lambda x: 2*x * hermite_polynomial(n - 1)(x) - 2 * (n - 1) * hermite_polynomial(n - 2)(x)


def solve_analytical_method(dx, x_max, n):
    """Returns values of the harmonic oscillator with energy 2*`n`+1 evaluated at points `x`.

    Parameters
    ----------
    dx : float
        spacing in the interval [0, x_max)
    n : int
        non-negative integer, the order of the polynomial
    x_max : float
        right end of the interval [0, x_max) at which solution is evaluated

    Returns
    -------
    ndarray
        x - values at which points are evaluated
    ndarray
        y(x) - values of the solution `y` evaluated at points in array `x`.
    """
    h = hermite_polynomial(n)
    normalisation_factor = round(h(0) if h(0) else derivative(h, 0))

    def hn(u):
        return h(u)/normalisation_factor

    x = np.arange(0, x_max, dx)
    y = np.vectorize(hn)(x)
    y *= np.exp(-x**2/2)
    return x, y


def solve_numerical_method(dx, x_max, n, e=None):
    if e is None:
        e = 2*n+1

    y0, dy0 = (n+1) % 2, n % 2

    def f(x):
        return x**2-e

    y1 = numerov_taylor_series(f=f, y0=y0, dy0=dy0, dx=dx, x0=0)

    y = numerov_delta(f=f, y0=y0, y1=y1, x_max=x_max, delta=dx)
    x = np.arange(0, x_max, dx)

    return x, y


def demonstrate_e_dependency():
    n = 0
    for e in [0.95, 1, 1.05]:
        x, y = solve_numerical_method(0.05, 5, n, e)
        plt.plot(x, y, label="Numerical, energy {}".format(e))

    xa, ya = solve_analytical_method(0.2, 5, n)
    plt.plot(xa, ya, ".", label="Analytical")

    plt.ylim(-1.1, 1.1)
    plt.title("Numerical stability as a function of energy")
    plt.xlabel(r"Position [a.u]")
    plt.ylabel(r"Solution [a.u]")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    demonstrate_e_dependency()
