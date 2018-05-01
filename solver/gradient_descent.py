from .derivatives import derivative


def gradient_descent(loss, x0=0, step=0.1, n_steps=10):
    """Calculates derivative of `f` of order `order` and evaluates it at point `x`

    Parameters
    ----------
    loss : func
        a function that takes float and returns float. Measures error that should be minimised
    x0 : float
        point at which neighborhood we expect a minimum
    step : float
        measure of step size. Scales gradient value
    n_steps : int
        how many steps should be performed

    Returns
    -------
    float
        point representing an approximation of a local minimum in neighborhood of x0
    float
        loss value at that point
    """
    for _ in range(n_steps):
        grad = derivative(loss, x=x0, order=1, dx=1e-2)
        x0 -= grad * step

    return x0, loss(x0)
