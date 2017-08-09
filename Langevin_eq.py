"""
This module contains the functions necessary for Velocity Verlet integration for a system that follows Langevin dynamics.
The system can have a number n_particles of particles in n_dim number of dimensions.
"""

import numpy as np



def new_half_velocity(current_vel, time_step, current_force, zeta, kB, Temp, atom_masses):
    """
    This function calculates the velocity of the particles in the system at time t + (delta t)/2.
    :param current_vel: a list of shape (n_particles, n_dim)
    :param time_step: float
    :param current_force: a list of shape (n_particles, n_dim)
    :param zeta: float
    :param kB: float
    :param Temp: float
    :param atom_masses: list of shape (n_particles,)
    :return: a list of shape (n_particles, n_dim)
    """
    dim = np.shape(current_vel)
    current_force = np.reshape(current_force,dim)
    new_half_vel = np.zeros(dim)


    for i in range(0, dim[0]):
        current_mass = atom_masses[i]
        root_term = np.sqrt(time_step * kB * Temp * zeta * (1 / current_mass))

        new_half_vel[i,:] = current_vel[i,:] + 0.5*time_step*((current_force[i,:]/current_mass) - zeta*current_vel[i,:]) + root_term * np.random.normal(0,1)

    return new_half_vel

def Langev_new_pos(current_pos, new_half_vel, time_step):
    """
    This function calculates the position of the particles in the system at a time t + delta t.
    :param current_pos: a list of shape (n_particles, n_dim)
    :param new_half_vel: a list of shape (n_particles, n_dim)
    :param time_step: float
    :return: a list of shape (n_particles, n_dim)
    """
    dim = np.shape(current_pos)
    new_pos = np.empty(dim)

    for i in range(0, dim[0]):
        new_pos[i,:] = current_pos[i,:] + new_half_vel[i,:]*time_step

    return new_pos

def Langev_new_vel(new_half_vel, new_force, zeta, kB, Temp, time_step, atom_masses):
    """
    This function calculates the velocity of the particles in the system at time t + delta t.
    :param new_half_vel: a list of shape (n_particles, n_dim)
    :param new_force: a list of shape (n_particles*n_dim,)
    :param zeta: float
    :param kB: float
    :param Temp: float
    :param time_step: float
    :param atom_masses: list of shape (n_particles,)
    :return:
    """
    dim = np.shape(new_half_vel)
    new_vel = np.empty(dim)
    new_force = np.reshape(new_force,dim)

    for i in range(0,dim[0]):
        current_mass = atom_masses[i]
        root_term = np.sqrt(time_step * kB * Temp * zeta * (1 / current_mass))
        new_vel[i,:] = new_half_vel[i,:] + 0.5*time_step*(new_force[i,:]*(1/current_mass) - zeta*new_half_vel[i,:]) + root_term*np.random.normal(0,1)

    return new_vel


# Comments

# Making the naming consistent: take "Langevin" away from the name
# Modify the shape of the force in the last function, so that all is consistent