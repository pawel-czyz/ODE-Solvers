======
Solver
======

A Python 3 package for solving computational physics problems. Implements:

 - Numerov ODE Solver
 - Runge-Kutta ODE Solver
 - 1D gradient descent optimisation
 - method for numerical derivatives calculation

Contains also various unit tests and examples, including:

 - 1D harmonic oscillator, solution in position basis
 - Lorentz attractor

Website: https://github.com/pawel-czyz/ODE-Solvers

Usage
-----

To install dependencies (Matplotlib and NumPy):

.. code:: bash

	$ make install

To run a harmonic oscillator example:

.. code:: bash

    $ make quantum

To run a Lorenz system simulation example:

.. code:: bash

	$ make lorenz

To run unit tests:

.. code:: bash

    $ make test

