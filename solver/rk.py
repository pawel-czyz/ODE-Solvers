import numpy as np


def rk(f, y0, t):
    """Function solving first-order ODEs using Runge-Kutta method.

    The ODE should have the form:
    y'(t) = f(y(t)), y(t0) = y0

    Parameters
    ----------
    f : func
        function that takes numpy array (k,) of coordinates and returns their derivatives as another array (k,)
    y0 : ndarray
        ndarray (k,)
    t : ndarray
        ndarray of shape (n,) with time points at which we want the value of function `y`

    Returns
    -------
    ndarray
        array with shape (n, k) with solution
    """

    solution = np.zeros((len(t), len(y0)))
    solution[0, :] = y0

    for i in range(1, len(solution)):
        dt = t[i] - t[i-1]

        yi0 = solution[i-1, :]
        fi0 = f(yi0)

        yi1 = yi0 + fi0*dt/2
        fi1 = f(yi1)

        yi2 = yi0 + fi1*dt/2
        fi2 = f(yi2)

        yi3 = yi0 + fi2*dt
        fi3 = f(yi3)

        yi4 = yi0 + (fi0 + 2*fi1 + 2*fi2 + fi3)*dt/6
        solution[i, :] = yi4

    return solution
