def derivative(f, x, order=1, dx=1e-3):
    """Calculates derivative of `f` of order `order` and evaluates it at point `x`

    Parameters
    ----------
    f : func
        a function that takes float and returns float
    x : float
        point at which the derivative should be evaluated
    order : 0, 1, 2
        order of the derivative
    dx : float
        difference used in calculation

    Returns
    -------
    float
        derivative of the function `f` evaluated at `x`
    """
    if order == 0:
        return f(x)
    elif order == 1:
        return (f(x+dx) - f(x-dx)) / (2*dx)
    elif order == 2:
        return (-f(x+2*dx) + 16*f(x+dx) - 30*f(x) + 16*f(x-dx)-f(x-2*dx)) / (12 * dx**2)
        # Rough approximation can be:
        # return (f(x+2*dx) - 2*f(x+dx) + f(x)) / dx**2
    else:
        raise ValueError("order can be 0, 1 or 2")


def numerov_taylor_series(f, y0, dy0, dx=1e-3, x0=0):
    """Uses Taylor series to approximate value:

    y(x0 + dx) = y(x0) + y'(x0) * dx + ...

    assuming y is the solution of the equation of the form:

    y''(x) = f(x) * y(x)

    Parameters
    ----------
    f : func
        a function that takes float and returns float
    y0 : float
        value of y(x0)
    dy0 : float
        value of y'(x0)
    dx : float
        step size
    x0 : float
        starting point at which values of y and y' are known

    Returns
    -------
    float
        value of y(x0+dx)
    """

    f0 = f(x0)
    df0 = derivative(f, x0, order=1, dx=dx)
    ddf0 = derivative(f, x0, order=2, dx=dx)

    return y0 + dx*dy0 + dx**2*f0*y0/2 + dx**3*(y0*df0 + dy0*f0)/6 + dx**4*(f0**2*y0 + 2*df0*dy0 + y0*ddf0)/24
