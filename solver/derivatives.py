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
        # return (f(x+2*dx) - 2*f(x+dx) + f(x)) / dx**2
    else:
        raise ValueError("order can be 0, 1 or 2")
