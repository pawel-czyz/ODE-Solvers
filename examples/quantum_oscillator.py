import sys
sys.path.append(".")
from solver.numerov import numerov_delta
from solver.derivatives import numerov_taylor_series, derivative


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


def hermite_polynomial_normalised(n):
    """Returns normalised Hermite polynomial of order `n`

    Parameters
    ----------
    n : int
        non-negative integer

    Returns
    -------
    func
        normalised Hermite polynomial `f` of order `n`, that is f(0) = 1 or f(0) = 0 and f'(0) = 1.

    Raises
    ------
    TypeError
        if `n` is not int
    ValueError
        if `n` is negative
    """
    h = hermite_polynomial(n)
    normalisation_factor = round(h(0) if h(0) else derivative(h, 0))
    return lambda x: h(x)/normalisation_factor


def solve_quantum_oscillator(dx, x_max, n, e):
    y0, dy0 = (n+1) % 2, n % 2

    def f(x):
        return x**2-e

    y1 = numerov_taylor_series(f=f, y0=y0, dy0=dy0, dx=dx, x0=0)

    print(numerov_delta(f=f, y0=y0, y1=y1, x_max=x_max, delta=dx))


if __name__ == "__main__":
    for n in range(8):
        h = hermite_polynomial_normalised(n)
        print(n, h(3), derivative(h, 0))

    # solve_quantum_oscillator(0.05, 5, 0, 1)
