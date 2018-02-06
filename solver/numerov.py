import numpy as np


def numerov(f, y0, y1, x):
    """Function solving second order ODEs using Numerov's method.

    The ODE should have the form:
    y''(x) = f(x) * y(x), y(x0) = y0, y(x1) = y1

    Parameters
    ----------
    f : func
        function that takes float and returns float being the value of f at this point
    y0 : float
        value of y at x[0]
    y1 : float
        value of y at x[1]
    x : ndarray
        array (n,) where n is greater or equal 2. Should have the position values at which y should be calculated.
        The points should be distributed uniformly (between any two in a row, the distance should be the same).

    Returns
    -------
    ndarray
        array (n,) where n is the length of x array, storing the values y(x)
    """

    # Initialise the solution array that will be returned
    solution = np.zeros_like(x)
    solution[0] = y0
    solution[1] = y1

    for i in range(2, len(solution)):
        dx = x[i] - x[i-1]

        rhs = (2 + 5*dx**2*f(x[i-1])/6) * solution[i-1] - (1 - dx**2*f(x[i-2])/12) * solution[i-2]
        lhs = (1 - dx**2*f(x[i])/12)

        solution[i] = rhs/lhs

    return solution


def numerov_delta(f, y0, y1, x_max, delta):
    """Function solving second order ODEs using Numerov's method.

    The ODE should have the form:
    y''(x) = f(x) * y(x), y(x0) = y0, y(x1) = y1

    Parameters
    ----------
    f : func
        function that takes float and returns float being the value of f at this point
    y0 : float
        value of y at x[0]
    y1 : float
        value of y at x[1]
    x_max : float
        the interval will be [0, x_max)
    delta : float
        spacing in the interval

    Returns
    -------
    ndarray
        array (n,) storing the values y(x), where x is the interval [0, x_max) spaced by delta

    Raises
    ------
    ValueError
        if delta/x_max is too large and the interval does not have enough points
    """

    x = np.arange(0, x_max, delta)

    if len(x) < 2:
        raise ValueError("delta is too big compared to x_max.")

    return numerov(f, y0, y1, x)
