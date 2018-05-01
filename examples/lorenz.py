import sys
sys.path.append(".")
from solver.rk import ode_solve_rk
import numpy as np
import matplotlib.pyplot as plt


def lorenz_equations(a, b, r):
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


def solve_lorenz(y0, a, b, r, t):
    """Solve Lorenz equations

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
    return ode_solve_rk(f=lorenz_equations(a, b, r), y0=y0, t=t)


def generate_plots(r, y0=(4, 5, 6), a=10, b=8/3, t=20):
    t_a = np.arange(0, t, 0.01)
    y = solve_lorenz(y0, a, b, r, t_a)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.axes().set_aspect('equal', 'datalim')
    f, axs = plt.subplots(1, 3)
    axs[0].plot(t_a, y[:, 0])
    axs[0].set_title(r"$y_0$ versus time")
    axs[1].plot(t_a, y[:, 1])
    axs[1].set_title(r"$y_1$ versus time")
    axs[2].plot(t_a, y[:, 2])
    axs[2].set_title("$y_2$ versus time")
    plt.savefig("ys_vs_time-r_{}.pdf".format(r))
    plt.gcf().clear()

    plt.plot(y[:, 1], y[:, 2])
    plt.xlabel(r"$y_1$")
    plt.ylabel(r"$y_2$")
    plt.savefig("y1_vs_y2-r_{}.pdf".format(r))
    plt.gcf().clear()
    #axs[1, 1].set_title('Phase diagram y_1 versus y_2')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    # plt.show()


def weather_forecasting():
    """Example of chaotic behaviour of the Lorenz system."""
    t = np.arange(0, 10, 0.01)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # --- Weather forecasting phenomenon ---
    ideal_start = np.array([4, 5, 6])

    peturbations = [0, 0.01, 0.1]  # [0, 0.01, 0.05]
    for peturbation in peturbations:
        y0 = ideal_start + peturbation

        y = solve_lorenz(y0 + peturbation, 10, 8 / 3, 28, t)[:, 0]
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
    for r in [1, 10, 28, 100]:
        print("Generating plot for r={}...".format(r))
        generate_plots(r)
    print("Weather forecasting...")
    weather_forecasting()
    print("Done.")
