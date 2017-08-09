"""
This module contains functions that are needed to calculate the forces and the energies of a 2D point particle in
a Muller Brown potential.
"""


import numpy as np

def MB_force(x, y):
    """
    This function calculates the force acting on a point particle with position (x,y) in a Muller Brown potential.
    :param x: float
    :param y: float
    :return: numpy array of shape (2,)
    """
    current_force = np.array([0.0, 0.0])

    # Parameters of the Muller-brown potential
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    xk = np.array([1, 0, -0.5, -1])
    yk = np.array([0, 0.5, 1.5, 1])

    # The force is calculated from the analytical differentiation of the Muller-Brown potential (with minus sign)

    for i in range(0,4):
        exp_arg = a[i]*(x-xk[i])**2 + b[i]*(x-xk[i])*(y-yk[i]) + c[i]*(y-yk[i])**2
        current_force[0] += -A[i] * np.exp(exp_arg) * (2*a[i]*(x-xk[i]) + b[i]*(y-yk[i]))
        current_force[1] += -A[i] * np.exp(exp_arg) * (2*c[i]*(y-yk[i]) + b[i]*(x-xk[i]))

    return current_force



def MB_potential(x,y):
    """
    This function calculates the energy of a point particle with position (x,y) in a Muller Brown potential.
    :param x: float
    :param y: float
    :return: float
    """

    # Parameters of the Muller-brown potential
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    xk = np.array([1, 0, -0.5, -1])
    yk = np.array([0, 0.5, 1.5, 1])

    # Calculating the value of the potential:
    MB_pot = 0

    for i in range(0,4):
        exp_arg = a[i]*(x-xk[i])**2 + b[i]*(x-xk[i])*(y-yk[i]) + c[i]*(y-yk[i])**2
        MB_pot += A[i] * np.exp(exp_arg)

    return MB_pot
