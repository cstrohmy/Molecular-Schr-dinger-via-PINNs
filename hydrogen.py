import numpy as np
from scipy.special import sph_harm, genlaguerre
from math import factorial, sqrt


# Exact solution
hydrogen_states = {}

def radial_wavefunction(n, l, r):
    """
    Spherically symmetric function of r, corresponds to electron shell n, orbital l
    :param n (int): electron shell
    :param l (int): orbital
    :param r (float): radius
    :return: the value of the radial part of the appropriate eigenfunction of the time-independent Schrödinger operator
    """
    rho = (2 * r) / n
    prefactor = sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    L = genlaguerre(n - l - 1, 2 * l + 1)
    radial_part = prefactor * np.exp(-rho / 2) * rho ** l * L(rho)
    return radial_part


# Iterate through all possible eigenstates up to and including the third electron shell, add them to hydrogen_states
max_num_shells = 3
for n in range(1, max_num_shells + 1):
    for l in range(n):
        for m in range(-l, l + 1):
            # Eigenvalue in atomic units
            eigenvalue = -1 / (2 * n ** 2)

            # Eigenstate (wave function)
            eigenstate = lambda r, theta, phi, n=n, l=l, m=m: radial_wavefunction(n, l, r) * sph_harm(m, l, phi, theta)

            # Store eigenvalue-eigenstate pair in hydrogen_states dictionary
            hydrogen_states[(n, l, m)] = (eigenvalue, eigenstate)


def laplacian(func, r, theta, phi, dr=1e-5, dtheta=1e-5, dphi=1e-5):
    """
    Computes laplacian of func in spherical coordinates using finite differences
    :param func: function to take laplacian of
    :param r: radius
    :param theta: xy angle
    :param phi: xy-z angle
    :param dr: r stepsize
    :param dtheta: theta stepsize
    :param dphi: phi stepsize
    :return:
    """
    # Radial derivative term
    radial_term = (func(r + dr, theta, phi) - 2 * func(r, theta, phi) + func(r - dr, theta, phi)) / dr ** 2 + \
                  (1 / r) * (func(r + dr, theta, phi) - func(r - dr, theta, phi)) / (2 * dr)

    # Theta derivative term
    theta_term = (func(r, theta + dtheta, phi) - 2 * func(r, theta, phi) + func(r, theta - dtheta, phi)) / dtheta ** 2 + \
                 (1 / (r * np.sin(theta))) * (func(r, theta + dtheta, phi) - func(r, theta - dtheta, phi)) / (
                             2 * dtheta)

    # Phi derivative term
    phi_term = (func(r, theta, phi + dphi) - 2 * func(r, theta, phi) + func(r, theta, phi - dphi)) / (
                r ** 2 * np.sin(theta) ** 2 * dphi ** 2)

    return radial_term + theta_term + phi_term


def verify_eigenstates(states_dict, num_samples=1000):
    """
    Checks that the functions collected in hydrogen_states are indeed eigenstates
    :param states_dict: dictionary consisting of eigenstates
    :param num_samples: number of points in domain R^3 to sample
    :return: mean square error of Hu - Eu, where H is our operator, u our eigenstate, and E the corresponding eigenvalue
    """
    errors = {}

    r_vals = np.random.uniform(.2, 8, num_samples)
    theta_vals = np.random.uniform(0, np.pi, num_samples)
    phi_vals = np.random.uniform(0, 2 * np.pi, num_samples)

    for key, value in states_dict.items():
        eigenvalue, eigenstate = value
        total_error = 0

        for r, theta, phi in zip(r_vals, theta_vals, phi_vals):
            lhs = -0.5 * laplacian(eigenstate, r, theta, phi) - eigenstate(r, theta, phi) / r
            rhs = eigenvalue * eigenstate(r, theta, phi)
            total_error += abs(lhs - rhs) ** 2

        errors[key] = total_error / num_samples

    return errors





















