import sys
sys.path.append(".")
from solver.rk import ode_solve_rk
import numpy as np
import matplotlib.pyplot as plt


def lorentz_equations(a, b, r):
    """Function preparing function returning time derivative of yi.

    Author
    ------
    Paweł Czyż, Date: 01/05/2018

    Parameters
    ----------
    a : float
        model constant
    b : float
        model constant
    r : float
        model constant

    Returns
    -------
    func
        function that takes ndarray (3,) of yi and returns ndarray (3,) of yi'
    """
    def f(y):
        return np.array([
            a * (y[1]-y[0]),
            r * y[0] - y[1] - y[0] * y[2],
            y[0] * y[1] - b * y[2]
        ])

    return f


def solve_lorentz(y0, a, b, r, t):
    """Solve Lorentz equations

    Parameters
    ----------
    y0 : ndarray
        ndarray with shape (3,) with initial value
    a : float
        model constant
    b : float
        model constant
    r : float
        model constant
    t : ndarray
        array (n, ) with times for which the system should be solved

    Returns
    -------
    ndarray
        array with shape (n, 3) with yi(t)
    """
    return ode_solve_rk(f=lorentz_equations(a, b, r), y0=y0, t=t)


def behaviour_changing_rayleigh_number():
    """Example of the behaviour change for different values of parameter `r`"""

    for i, r in enumerate([0.01, 1, 2, 3, 4]):
        t = np.arange(0, 12, 0.01)
        y = solve_lorentz(np.array([4, 5, 6]), 10, 8/3, r, t)

        ax0 = plt.subplot2grid((5, 4), (i, 0))
        ax0.plot(y[:, 0], y[:, 1])

        ax1 = plt.subplot2grid((5, 4), (i, 1))
        ax1.plot(y[:, 0], y[:, 2])

        ax2 = plt.subplot2grid((5, 4), (i, 2))
        ax2.plot(y[:, 1], y[:, 2])

        ax3 = plt.subplot2grid((5, 4), (i, 3))
        ax3.plot(t, y[:, 0])

    plt.axis('off')

    plt.show()
    #
    #     if i % 4 == 0:
    #         axs[i].plot(y[:, 0], y[:, 1])
    #     elif i % 4 == 1:
    #         axs[i].plot(y[:, 1], y[:, 2])
    #     elif i % 4 == 2:
    #         axs[i].plot(y[:, 2], y[:, 0])
    #     else:
    #         axs[i].plot(t, y[:, 0])
    #
    #     axs[i].set_title("r={}".format(r))
    #
    # plt.show()


def weather_forecasting():
    """Example of chaotic behaviour of the Lorentz system."""
    t = np.arange(0, 10, 0.01)

    # --- Weather forecasting phenomenon ---
    ideal_start = np.array([4, 5, 6])

    peturbations = [0, 0.01, 0.1]  # [0, 0.01, 0.05]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for peturbation in peturbations:
        y0 = ideal_start + peturbation

        y = solve_lorentz(y0+peturbation, 10, 8/3, 28, t)[:, 0]
        plt.plot(t, y, label="{}".format(peturbation))

    plt.title("Weather forecasting")
    plt.xlabel(r"t [a.u]")
    plt.ylabel(r"y_0 [a.u]")
    plt.legend(title="Perturbation")
    # plt.show()
    plt.savefig("Weather_forecasting.pdf")
    # Clear Matplotlib memory
    plt.gcf().clear()

if __name__ == "__main__":
    # behaviour_changing_rayleigh_number()
    weather_forecasting()
