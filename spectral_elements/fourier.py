import numpy as np
from math import floor, pi, sin, tan


def DiscreteFourierCoefficients(func_evaluation):
    """Direct evaluation of the discrete Fourier coefficients.
    Algorithm 1 in Kopriva (2009).

    Args:
        func_evaluation (list): Values of the function to be evaluated.

    Returns:
        list: List of the Fourier coefficients.
    """
    i = 1j

    # Number of points
    n = len(func_evaluation)
    k_min = -floor(n / 2)
    k_max = floor(n / 2)

    # Initialize the list of coefficients
    fourier_coefficients = []

    # Loop over the coefficients
    for k in range(k_min, k_max + 1):
        # Initialize the sum
        sum = 0

        # Loop over the points
        for j in range(0, n):
            sum += func_evaluation[j] * np.exp(-2.0 * pi * i * j * k / n)

        # Append the coefficient
        fourier_coefficients.append(sum / n)

    return fourier_coefficients


def FourierInterpolantFromModes(fourier_coefficients, x):
    """Interpolant of a function from its Fourier coefficients.
    Algorithm 2 in Kopriva (2009).

    Args:
        fourier_coefficients (list): List of Fourier coefficients.
        x (list): List of points where the interpolant is evaluated.

    Returns:
        list: List of values of the interpolant.
    """
    i = 1j

    # Initialize the list of values
    values = []

    n = len(fourier_coefficients)
    k_min = -floor(n / 2)
    k_max = floor(n / 2)
    K = np.linspace(k_min, k_max, n)

    # Loop over the points
    for x_i in x:
        # Initialize the sum
        sum = (fourier_coefficients[0] * np.exp(-i * K[0] * x_i) +
               fourier_coefficients[-1] * np.exp(i * K[-1] * x_i)) / 2

        # Loop over the coefficients
        for k_i in range(1, len(K) - 1):
            k = K[k_i]
            fk = fourier_coefficients[k_i]

            # Sum the terms
            sum += fk * np.exp(i * k * x_i)

        # Append the value
        values.append(np.real(sum))

    return values


def FourierInterpolantFromNodes(x, nodes, func_evaluation):
    """Interpolant of a function from its nodes.
    Algorithm 3 in Kopriva (2009).

    Args:
        x (list): List of points where the interpolant is evaluated.
        nodes (list): List of nodes.
        func_evaluation (list): List of values of the function to be evaluated.

    Returns:
        list: List of values of the interpolant.
    """

    # Initialize the list of values
    values = []

    # Number of nodes
    n = len(nodes)

    if n % 2 != 0:
        raise ValueError('The number of nodes must be even.')

    # Loop over the points
    for x_i in x:
        # Test if the point is a node
        is_node = False
        for j in range(n):
            x_j = nodes[j]
            if abs(x_i - x_j) < 1e-12:
                values.append(func_evaluation[j])
                is_node = True
                break

        if is_node:
            continue

        # Initialize the sum
        sum = 0

        # Loop over the nodes
        for j in range(n):
            x_j = nodes[j]

            # Sum the terms
            t = 0.5 * (x_i - x_j)
            sum += func_evaluation[j] * sin(n * t) / tan(t)

        # Append the value
        values.append(sum / n)

    return values
