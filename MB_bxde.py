"""
This script simulates a 2D point particle in a Muller Brown potential with BXD constraints in energy space.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import muller_brown
import bxd
import Langevin_eq
plt.rcParams['animation.ffmpeg_path'] = '/Users/walfits/anaconda3/envs/deffi/bin/ffmpeg' # Change this to ffmpeg binary


def update(frame, line_main, line_history, trajectory, ax):
    """
    This function tells the animation.FuncAnimation() function how to update the animation at a certain frame.
    The line_main object contains the current position of the system, while line_history contains the time history of
    where it has been.
    :param frame: int
    :param line_main: object belonging to the matplotlib.lines.Line2D class
    :param line_history: object belonging to the matplotlib.lines.Line2D class
    :param trajectory: list of shape (2*num_steps)
    :param ax: the axes of the figure being drawn
    :return: line_main - object belonging to the matplotlib.lines.Line2D class
    """

    current_pos = [trajectory[2*frame]], [trajectory[2*frame+1]]

    hist_x = []
    hist_y = []
    for l in range(0,frame-1,5):
        hist_x.append(trajectory[2*l])
        hist_y.append(trajectory[2*l+1])

    line_main.set_data(current_pos[0], current_pos[1])  # This shows the main particle
    line_history.set_data(hist_x, hist_y)               # This shows the time history of the particle positions

    return line_main,

def update_grey(frame, line_main, line_history, trajectory, ax):
    """
    This function tells the animation.FuncAnimation() function how to update the animation at a certain frame.
    The line_main object contains the current position of the system, while line_history contains the time history of
    where it has been. It also adds grey areas to the zones that the particle has visited and can't visit anymore.

    To increase the number of grey zones that appear add more elements to the 'change' list in the same way that they have
    been added up to now.

    :param frame: int
    :param line_main: object belonging to the matplotlib.lines.Line2D class
    :param line_history: object belonging to the matplotlib.lines.Line2D class
    :param trajectory: list of shape (2*num_steps)
    :param ax: the axes of the figure being drawn
    :return: line_main - object belonging to the matplotlib.lines.Line2D class
    """

    change = np.zeros(3)
    change[0] = final_times[0] - final_times[0] % increment
    change[1] = (final_times[0] + final_times[1]) - (final_times[0] + final_times[1]) % increment
    change[2] = (final_times[0] + final_times[1] + final_times[2]) - (final_times[0] + final_times[1] + final_times[
        2]) % increment

    if frame == change[0]:    # Change this number with the frame number at which a grey area should appear
        bxd.mixed_contour(sep_bound, contour_lines, contour_lines[1], ax)
    elif frame == change[1]:
        bxd.mixed_contour(sep_bound, contour_lines, contour_lines[2], ax)
    elif frame == change[2]:
        bxd.mixed_contour(sep_bound, contour_lines, contour_lines[3], ax)

    current_pos = [trajectory[2 * frame]], [trajectory[2 * frame + 1]]

    hist_x = []
    hist_y = []
    for l in range(0, frame - 1, 5):
        hist_x.append(trajectory[2 * l])
        hist_y.append(trajectory[2 * l + 1])

    line_main.set_data(current_pos[0], current_pos[1])  # This shows the main particle
    line_history.set_data(hist_x, hist_y)  # This shows the time history of the particle positions

    return line_main,

# Set up formatting for the movie files
writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Parameters needed
num_steps = 4000
time_step = 0.001
mass = [1]
zeta = 10  # Friction parameter for Langevin Thermostat - zeta=0 gives NVE
temp = 15
kb = 1  # kB in atomic units is 1 by definition

# Initialising position and velocities
current_pos = np.array([0.6, 0]).reshape((1,2))
current_vel = np.array([-13, -15]).reshape((1,2))
current_force = muller_brown.MB_force(current_pos[0,0], current_pos[0,1]).reshape((1,2))
trajectory = current_pos[0,:]        # This 1D vector will contain the whole trajectory

# Defining the BXD energy boundaries and the maximum number of impacts with a boundary
global sep_bound, contour_lines
current_pot = muller_brown.MB_potential(current_pos[0,0], current_pos[0,1])
sep_bound = 50               # energy separation of the boxes
initial_box = current_pot - sep_bound * 0.5
high_bound, low_bound = bxd.BXD_boundaries(current_pot, sep_bound, initial_box)
max_impact = 30

# Defining the counters
low_counter = 0                         # These count the impacts with the boundaries of a box
high_counter = 0
final_counts_high = []         # These will contain the overall count of impacts per boundary
final_counts_low = []
time_counter = 0                        # This counts the time-steps in each box
global final_times
final_times = []              # This will contain the counts for each box

# Preparing for plotting the potential (background to the animation)
x_range = np.arange(-2.5, 1.4, 0.001)
y_range = np.arange(-0.7, 2.5, 0.001)
X, Y = np.meshgrid(x_range, y_range)
Z = muller_brown.MB_potential(X,Y)

# Plotting the potential and the particle (this is a 'line', otherwise the animation function won't accept it)
contour_lines = bxd.BXD_contour(sep_bound, initial_box)
fig = plt.figure()
ax = plt.axes()
back_fig = ax.contourf(X, Y, Z, 50, alpha=.75, cmap='rainbow', extend="both", levels= contour_lines)
# plt.colorbar(back_fig)
part_fig, = ax.plot(current_pos[0,0], current_pos[0,1], ls='None', lw=1, color='blue', marker='o', ms=8, alpha=1)
history_fig, = ax.plot([], [], lw=1.5, color='#1f537a', alpha=0.4)  # for dotted history add :ls="None", marker="o", mew=0.0

# Velocity verlet with bxd
for i in range(0,num_steps):

    # Incrementing time step counter
    time_counter += 1

    new_half_vel = Langevin_eq.new_half_velocity(current_vel, time_step, current_force, zeta, kb, temp, mass)
    new_pos = Langevin_eq.Langev_new_pos(current_pos, new_half_vel, time_step)
    new_force = muller_brown.MB_force(new_pos[0,0], new_pos[0,1])
    new_vel = Langevin_eq.Langev_new_vel(new_half_vel, new_force, zeta, kb, temp, time_step, mass)
    new_pot = muller_brown.MB_potential(new_pos[0,0], new_pos[0,1])

    # BXD constraints are applied
    if new_pot <= low_bound:
        new_pos = current_pos
        # new_force = current_force
        new_vel = bxd.BXD_vel_inv(current_force,current_vel,mass)
        low_counter += 1
    elif new_pot >= high_bound:
        if high_counter < max_impact:
            new_pos = current_pos
            # new_force = current_force
            new_vel = bxd.BXD_vel_inv(current_force, current_vel, mass)
            high_counter += 1
        elif high_counter == max_impact:
            final_counts_low.append(low_counter)
            low_counter = 0
            final_counts_high.append(high_counter)
            high_counter = 0
            final_times.append(time_counter)
            time_counter = 0
            low_bound += sep_bound
            high_bound += sep_bound

    # Updating force, velocity and position
    current_pos = new_pos
    current_vel = new_vel
    current_force = new_force
    trajectory = np.concatenate((trajectory,current_pos[0,:]),axis=0)

# Uncomment the following lines to output information for postprocessing to a file
# import output
# output.write_kinetics(final_counts_high, final_counts_low, final_times, time_step)

# Animation
global increment
increment = 20 # Increasing the increment speeds up the animation
frames = list(range(0, num_steps, increment))
ani = animation.FuncAnimation(fig, update_grey, frames, init_func=None, blit=False, interval=1, fargs=(part_fig, history_fig, trajectory, ax))
ani.save('BXD_1.mp4', writer=writer, dpi=200) # Uncomment to save the animation in mp4 format
plt.show()