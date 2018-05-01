from .derivatives import derivative

DELTA_MINIMAL = 1e-2
DELTA_END = 1e-5


def _sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    return 0


def gradient_descent(loss, x0=0, step=0.1, n_steps=10, dx=1e-2):
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
        grad = (loss(x0 + dx) - loss(x0)) / dx

        delta = - grad * step

        # If step is changes nothing, we can stop the iteration process
        if abs(delta) < DELTA_END:
            break
        # If step should change something, but the convergence would be too slow, we artificially inscrease it
        if abs(delta) < DELTA_MINIMAL:
            delta = _sign(delta) * DELTA_MINIMAL
        # Make step
        x0 += delta

    return x0, loss(x0)
