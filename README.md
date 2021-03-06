# Simulation of model systems with boxed molecular dynamics

![](https://github.com/SilviaAmAm/bxde/blob/master/visuals/bxd.gif)

This repository contains the following scripts:

1. [MB_unbiased.py](./MB_unbiased.py):
This script plots an animation of a simulation of a point particle in a model 2D Muller-Brown potential. It can be run in NVE or NVT ensembles.

2. [MB_bxde.py](./MB_bxde.py):
This script plots an animation of a simulation of a point particle in a model 2D Muller-Brown potential with BXD in energy space. It can be run in NVE or NVT.

The main functionality is implemented in:

1. [bxd.py](./bxd.py):
This contains the function that generate the BXD boundaries and the function that does the velocity inversion.

2. [Langevin_eq.py](./Langevin_eq.py):
This contains the functions that are required to do the Velocity Verlet algorithm with the Langevin equation of motion.

3. [muller_brown.py](./muller_brown.py):
This contains the functions needed to calculate the forces and the potential energy acting on a 2D point particle in a model Muller-Brown potential.

4. [output.py](./output.py):
This contains the function that enable to write data from the simulation to a file.

## How to use the scripts

The interpreter used was python 3.6

Both scripts depend on the following packages:
1. Numpy
2. Matplotlib
3. FFMpeg (optional)

In order to be able to save the animations, here ffmpeg is used. This can be installed using Anaconda. See this link: https://anaconda.org/menpo/ffmpeg

### MB_unbiased.py

The parameters of the simulation, such as the temperature, the number of steps, the size of the time step, can be changed in this section of the code:

```python
# Parameters needed
num_steps = 4000
time_step = 0.001
mass = [1]
zeta = 5  # Friction parameter for Langevin Thermostat, set to 0 for NVE simulation
Temp = 15  # Temperature K
kB = 1  # Boltzmann constant
```

In addition, to run a NVE simulation, set `zeta = 0`. Otherwise, set to a suitable value for NVT simulation.

If you want to save the animations, there are a few things to check. You may have to add this line at the top of the script:

```python
plt.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg'
```

so you will have to change the path to where your ffmpeg binary lives. Then, you need to make sure that this line at the bottom of the script is uncommented:

```python
ani.save(filename='MD.mp4', writer=writer, dpi=200)
```

### MB_bxde.py

The same parameters as for MB_unbiased.py can be modified. In addition, the separation between the bxd boundaries in energy space can be changed by modifying the value of `sep_bound`:

```python
# Defining the BXD energy boundaries and the maximum number of impacts with a boundary
global sep_bound, contour_lines
current_pot = muller_brown.MB_potential(current_pos[0,0], current_pos[0,1])
sep_bound = 30               # energy separation of the boxes
initial_box = current_pot - sep_bound * 0.5
high_bound, low_bound = bxd.BXD_boundaries(current_pot, sep_bound, initial_box)
max_impact = 20
```

The maximum number of impacts with the higher boundary can be specified through `max_impact`.

If the particle does not explore at least 3 boxes, the update_grey function will give an error. If this happens, increase the number of time steps that he simulation is run for.
