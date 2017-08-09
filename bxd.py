"""
This module contains functions specific to the bxd procedure.
"""

import bisect
import numpy as np

def BXD_boundaries(curr_pot, sep_bound, initial_box):
    """
    This function creates a set of boundaries in energy space based on the value of the current potential energy value,
    the user defined separation between the boundaries, and the energy value of the first boundary. It then returns the
    value of the current high an low boundaries of the system.
    :param curr_pot: float
    :param sep_bound: float
    :param initial_box: float
    :return: float, float
    """

    the_boxes = [initial_box, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          # Creates a list of boundaries and check in which box is the particle
    for i in range(0,len(the_boxes)):
        the_boxes[i] = the_boxes[0] + sep_bound*i
    index = bisect.bisect(the_boxes,curr_pot)
    high_bound = the_boxes[index]
    low_bound = the_boxes[index-1]

    return high_bound, low_bound

def BXD_vel_inv(Dphi, curr_vel, atom_masses):
    """
    This function calculates the inverted velocity when a particle hits a boundary.
    :param Dphi: derivative of the collective variable. Here the collective variable is the energy, so Dphi is the
                 force. It has shape (n_particles*n_dim,)
    :param curr_vel: a list of shape (n_particles, n_dim)
    :param atom_masses: a list of shape (n_particles, )
    :return: a list of shape (n_particles, n_dim)
    """

    dim = np.shape(curr_vel)
    new_vel = np.zeros(dim)
    Mass_matrix = np.zeros([dim[1]*dim[0],dim[1]*dim[0]])

    for i in range(0,dim[0]):
        current_mass = atom_masses[i]
        for j in range(0, dim[1]):
            Mass_matrix[3*i + j, 3*i + j] = current_mass

    # For BXD in energy, Dphi is the force and it is a 1x3N vector.  invM_DphiT will be a 3Nx1 vector
    invM_DphiT = np.dot(np.linalg.inv(Mass_matrix), np.matrix.transpose(Dphi))   # Refers to the M^-1 * (Del Phi)^T

    numerator = -2 * np.dot(Dphi, curr_vel.flatten())
    denominator = np.dot(Dphi,invM_DphiT)
    lambda_multip = numerator/denominator

    new_vel = curr_vel + lambda_multip * np.reshape(invM_DphiT,dim)

    return new_vel

def BXD_contour(sep_bound, initial_box):
    """
    This function generates a list of all the box boundaries from the value of the initial boundary and the separation
    between the boundaries.
    :param sep_bound: float
    :param initial_box: float
    :return: list of shape (12,)
    """

    the_boxes = [initial_box, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          # Creates a list of boundaries and check in which box is the particle
    for i in range(0,len(the_boxes)):
        the_boxes[i] = the_boxes[0] + sep_bound*i

    return the_boxes

def grayify_cmap(cmap):
    """
    This function turns a colour map in a gray scale version of it.
    :param cmap: string - name of the matplotlib colourmap
    :return: the grayfied colour map
    """
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)

def mixed_contour(sep_bound, contour_lines, grey_bound, ax):
    """
    This function gives back a contour where only the region below a certain number are coloured in grey
    :param sep_bound: float
    :param contour_lines: list of shape (n_boundaries,)
    :param grey_bound: float - threshold that below which everything is coloured in a grey scale
    :param ax:
    :return:
    """
    import muller_brown
    from numpy.ma import masked_array

    x_range = np.arange(-2.5, 1.4, 0.0025)
    y_range = np.arange(-0.7, 2.5, 0.0025)
    X, Y = np.meshgrid(x_range, y_range)
    Z = muller_brown.MB_potential(X,Y)
    new_Z = masked_array(Z,Z >= grey_bound)       # This creates an array where all the values above the grey bound are masked
    cbb = ax.contourf(X, Y, new_Z, alpha=.75, cmap=grayify_cmap("rainbow"), extend="both", levels=contour_lines)

    return cbb