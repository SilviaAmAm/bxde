"""
This script does a simulation of a 2D point particle in a Muller Brown potential in the NVE or NVT ensemble.
The dynamics is unbiased.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import muller_brown
import Langevin_eq

def update(frame, line_main, line_history, trajectory):
    """
    This function tells the animation.FuncAnimation() function how to update the animation at a certain frame.
    The line_main object contains the current position of the system, while line_history contains the time history of
    where it has been.
    :param frame: int
    :param line_main: object belonging to the matplotlib.lines.Line2D class
    :param line_history: object belonging to the matplotlib.lines.Line2D class
    :param trajectory: list of shape (2*num_steps)
    :return: line_main - object belonging to the matplotlib.lines.Line2D class
    """

    current_pos = [trajectory[2*frame]],[trajectory[2*frame+1]]

    hist_x = []
    hist_y = []
    for l in range(0,frame-1,5):
        hist_x.append(trajectory[2*l])
        hist_y.append(trajectory[2*l+1])

    line_main.set_data(current_pos[0], current_pos[1])  # This shows the main particle
    line_history.set_data(hist_x, hist_y)               # This shows the time history of the particle positions

    return line_main,


# Set up formatting for the movie files
the_writer = animation.writers['ffmpeg']
writer = the_writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Parameters needed
num_steps = 4000
time_step = 0.001
mass = [1]
zeta = 5  # Friction parameter for Langevin Thermostat, set to 0 for NVE simulation
Temp = 15  # Temperature K
kB = 1  # Boltzmann constant


# Initialising position and velocities
current_pos = np.array([0.3, 0.4])
current_vel = np.array([-0.5, 0.5])
dim = current_vel.shape
current_force = muller_brown.MB_force(current_pos[0], current_pos[1])
trajectory = current_pos

# Preparing for plotting the potential (background to the animation)
x_range = np.arange(-1.7, 1.2, 0.01)
y_range = np.arange(-0.5, 2.25, 0.01)
X, Y = np.meshgrid(x_range, y_range)
Z = muller_brown.MB_potential(X,Y)

# Plotting the potenital and the particle (this is a 'line', otherwise the animation function won't accept it)
fig = plt.figure()
sep_bound = 25 # Separation between the lines of the contour
contour_lines = np.arange(-200.0, 400, sep_bound)
back_fig = plt.contourf(X, Y, Z, 50, alpha=.75, cmap='rainbow', extend="both", levels=contour_lines)
part_fig, = plt.plot(current_pos[0], current_pos[1], ls='None', lw=1.0, color='blue', marker='o', ms=8, alpha=1)
history_fig, = plt.plot([], [], lw=2.0, color='#2c70a3', alpha=0.5)

# Velocity verlet
for i in range(0,num_steps):

    new_half_vel = Langevin_eq.new_half_velocity(current_vel.reshape((1,2)), time_step, current_force, zeta, kB, Temp, mass)
    new_pos = Langevin_eq.Langev_new_pos(current_pos.reshape((1,2)), new_half_vel, time_step)
    new_force = muller_brown.MB_force(new_pos[0][0], new_pos[0][1])
    new_vel = Langevin_eq.Langev_new_vel(new_half_vel.reshape((1,2)), new_force, zeta, kB, Temp, time_step, mass)

    # Updating force, velocity and position

    current_pos = new_pos[0]
    current_vel = new_vel[0]
    current_force = new_force
    trajectory = np.concatenate((trajectory,current_pos),axis=0)


frames = list(range(0, num_steps, 5))
ani = animation.FuncAnimation(fig, update, frames, init_func=None, blit=False, interval=1, fargs=(part_fig, history_fig, trajectory))
# ani.save('MD_NVE.mp4', writer=writer) # Uncomment this to save the animation as a mp4 file
plt.show()
