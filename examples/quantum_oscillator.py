import sys
sys.path.append(".")
from solver import ode_solve_numerov, ode_solve_numerov_delta, numerov_taylor_series, derivative, gradient_descent
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

    y = ode_solve_numerov_delta(f=f, y0=y0, y1=y1, x_max=x_max, delta=dx)
    x = np.arange(0, x_max, dx)

    return x, y


def demonstrate_e_dependency(n=0):
    e_mean = 2*n+1
    for e in [e_mean-0.05, e_mean, e_mean+0.05]:
        x, y = solve_numerical_method(0.05, 5, n, e)
        plt.plot(x, y, label="Numerical, energy {}".format(e))

    xa, ya = solve_analytical_method(0.2, 5, n)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot analytic solution
    plt.plot(xa, ya, ".", label="Analytical")

    plt.ylim(-1.1, 1.1)
    plt.title("Numerical stability as a function of energy")
    plt.xlabel("Position [a.u]")
    plt.ylabel("Solution [a.u]")
    plt.legend()
    # plt.show()
    plt.savefig("Quantum_oscillator-{}.pdf".format(n))
    # Clear Matplotlib memory
    plt.gcf().clear()

# Implementation of the required API


def solve_numerov(f, x, psi0, dpsi0):
    """Solving psi''(x) = f(x) * psi(x), with psi(x0)=psi0, psi'(x0)=dpsi0

    Author
    ------
    Paweł Czyż, Date: 01/05/2018

    Parameters
    ----------
    f: func
        function in the ODE. Should take float as a value and return float.
    x: ndarray
        domain of integgration. The first element x[0] is value x0 at which boundary conditions are given
    psi0 : float
        psi(x0)
    dpsi0 : float
        psi'(x0)

    Returns
    -------
    ndarray
        values of psi evaluated in points of the array x

    Example
    -------
    >>> def f(x):
    ... return x**2-1
    >>> x = np.linspace(0, 5, 100)
    >>> psi0 = 1
    >>> dpsi0 = 0
    >>> psi = solve_numerov(f, x, psi0, dpsi0)
    """

    psi1 = numerov_taylor_series(f=f, y0=psi0, dy0=dpsi0, dx=x[1]-x[0], x0=x[0])
    psi = ode_solve_numerov(f=f, y0=psi0, y1=psi1, x=x)
    return psi


def find_oscillator_eigenvalue(e0):
    """Finding the eigenvalue for harmonic oscillator potential by iterative gradient descent method.

    Author
    ------
    Paweł Czyż, Date: 01/05/2018

    Parameters
    ----------
    e0 : float
        positive number being an initial guess of the eigenvalue

    Returns
    -------
    float
        the eigenvalue near the initial guess of the system with harmonic oscillator

    Example
    -------
    >>> print(find_oscillator_eigenvalue(1.2))
    0.98
    >>> print(find_oscillator_eigenvalue(3.4))
    3.0
    """
    n = int(0.5 * (e0 - 1))

    def loss(et):
        x, y = solve_numerical_method(0.05, 5, n, et)
        return (y[-1]/100)**2

    e_found, _ = gradient_descent(loss, e0, step=2e-5, dx=0.05, n_steps=200)
    return e_found

if __name__ == "__main__":
    # Producing plots
    print("Plots are being generated...")
    for n in [0, 1, 2, 3, 7]:
        print("Generating {}-quanta oscillator plot...".format(n))
        demonstrate_e_dependency(n)
    print("Finished plots.")

    # Finding eigenvalues from initial guess
    for e in [0.8, 1.2, 1.4, 3.4, 5.7]:
        print("Finding eigenvalue by iteration starting from {}...".format(e))
        print("Found {:.2}.".format(find_oscillator_eigenvalue(e)))
