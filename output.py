"""
This module contains function that write the output of the simulations to a file for analysis
"""

import numpy as np

def write_kinetics(final_count_high, final_count_low, final_times, time_step):
    """
    This function generates a file "kinetics.txt" with the kinetics data obtained during a simulation. final_count_high
    and contains the number of impacts with each boundary when they are the higher energy boundary and final_count_low
    contains the number of impacts with each boundary when they are the lower energy boundary. final_times contains the
    number of time steps spent in each box.
    :param final_count_high: list of shape (n_boundaries,)
    :param final_count_low: list of shape (n_boundaries,)
    :param final_times: list of shape (n_boundaries,)
    :param time_step: float
    :return: none
    """

    # Opening file to write results
    f1 = open('kinetics.txt', 'w+')

    # Making a table into the result file
    size = len(final_count_high)

    seq = ["box", "final count high", "final count low", "final times in box"]
    seq1 = '{0[0]:>10}{0[1]:>20}{0[2]:>20}{0[3]:>25}'.format(seq)
    f1.writelines(seq1)
    f1.write("\n")

    for i in range(0,size):
        seq = [str(i), str(final_count_high[i]), str(final_count_low[i]), str(final_times[i])]
        seq1 = '{0[0]:>10}{0[1]:>20}{0[2]:>20}{0[3]:>25}'.format(seq)
        f1.writelines(seq1)
        f1.write("\n")

    f1.write("\ntime step: " + str(time_step))
    f1.close()

    return None


def write_energy(pot_energy, kin_energy, temperature):
    """
    This function writes to a file "energy.txt" the value of the potential and kinetic energy and the temperature.
    The units depend on the force/potential energy routines that are passed.
    :param pot_energy: float
    :param kin_energy: float
    :param temperature: float
    :return: None
    """

    # Opening file to write results
    f2 = open('energy.txt', 'w+')

    seq = ["Potential energy", "Kinetic energy", "Total energy", "Temperature"]
    seq1 = '{0[0]:>10}{0[1]:>20}{0[2]:>20}{0[3]:>20}'.format(seq)
    f2.writelines(seq1)
    f2.write("\n")

    size = len(pot_energy)

    for i in range(0,size):
        seq = [str(pot_energy[i]), str(kin_energy[i]), str(pot_energy[i]+kin_energy[i]), str(temperature[i])]
        seq1 = '{0[0]:>10}{0[1]:>20}{0[2]:>20}{0[3]:>20}'.format(seq)
        f2.writelines(seq1)
        f2.write("\n")

    f2.close()

    return None

def write_trajectory(trajectory, atoms, dim):
    """
    This function writes the trajectory of the simulation to a file "trajectory.xyz" in a format that is readable by VMD.
    :param trajectory: list of shape (n_particles * n_dim * n_time_steps,)
    :param atoms: list of characters of shape (n_particles,) where each element is the chemical symbol of an atom.
    :param dim: tuple (n_particles, n_dim)
    :return: None
    """

    dim_traj = np.shape(trajectory)
    traj_file = open('trajectory.xyz', 'w+')

    for j in range(0, dim_traj[0], 5):
        current_pos = np.reshape(trajectory[j,:], dim)
        traj_file.write(str(dim[0]) + "\n")
        traj_file.write("\n")

        for i in range(0, dim[0]):
            seq = [atoms[i], str(current_pos[i,0]), str(current_pos[i,1]), str(current_pos[i,2])]
            seq1 = '{0[0]:>10}{0[1]:>20}{0[2]:>20}{0[3]:>20}'.format(seq)

            traj_file.writelines(seq1)
            traj_file.write("\n")

    traj_file.close()
