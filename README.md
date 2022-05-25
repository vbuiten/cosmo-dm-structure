# cosmo-dm-structure
Cosmological Physics 2022 Project 3: Cosmological dark matter simulation using the Particle Mesh method

Author: Victorine Buiten (buiten@strw.leidenuniv.nl).

Date last updated: 25/05/2022

## What's in this repository?

This repository contains code for simulating a set of dark matter particles against a background universe governed by the Friedmann equation.
The code uses the Particle Mesh method, combining individual particles with a grid.
The code is structured as follows:
* _framework_ contains the code required for the set-up of the particles and the grid. The ParticleMesh class allows the particles and grid to interact.
* _simulation_ contains the code for running the simulations. Simulations are run through the Simulator class, which uses a leapfrog integration scheme.
* _data_ contains the code for loading the data of previously-run simulations, and for animating a 3D simulation.
* _analysis_ contains the code for computing and plotting pair correlation functions from evolved simulations. Specifically, the Landy-Szalay estimator (1993) is used (https://ui.adsabs.harvard.edu/abs/1993ApJ...412...64L/abstract).
* _tests_ contains a number of test files that were used in testing the code during development, and for running series of simulations.
* main.py is a main script that can be used to run a simulation, make an animation of it and plot the correlation function of the final state of the simulation. It also serves as an example script for users of the code.

## What are the prerequisites?

The simulations use NumPy, Numba and SciPy. For saving the simulation data, the HDF5 format is used, and thus h5py is required. The plotting code demands matplotlib.
The animation code requires having FFmpeg installed.

## How to run the code?
For a complete example, see the script main.py:

```
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.simulator import Simulator
from data.animation import Animation3D
from analysis.correlation import CorrelationFunction

# set up the size of the box and number of particles
linear_size = 20
n_particles = 16**3
dim = 3

# create the grid and particle set
grid = Grid(linear_size, dim)
particles = ParticleSet(linear_size, dim, n_particles)
particles.uniformRandomPositions()
particles.zeroMomenta()
pm = ParticleMesh(grid, particles, method="CIC")
pm.densityFromParticles()

# specify the relevant scale factors and scale factor steps
a_start = 1e-2
a_step = 1e-4
a_step_save = 1e-3
a_end = 3e-2

# specify the cosmological model
matter_dens = 0.3
lambda_dens = 0.7
curv_dens = 0.

# run and save the simulation
simfile = "size{}_N{}_Om0_{}_Ode0_{}_Ok0_{}.hdf5".format(linear_size, n_particles, matter_dens, lambda_dens, curv_dens)
sim = Simulator(pm, a_start, a_step, matter_dens, lambda_dens, curv_dens)
sim.evolve(a_end, simfile, save_step=a_step_save)

print ("Finished the simulation.")

# create an animation
animfile = "size{}_N{}_Om0_{}_Ode0_{}_Ok0_{}.mp4".format(linear_size, n_particles, matter_dens, lambda_dens, curv_dens)
animation = Animation3D(simfile)
animation.animate()
animation.save(animfile)

# compute and plot the correlation function
corrfile = "size{}_N{}_Om0_{}_Ode0_{}_Ok0_{}_corr_func.png".format(linear_size, n_particles, matter_dens, lambda_dens,
                                                                    curv_dens)
corr = CorrelationFunction(simfile)
corr.plot()
corr.show()
corr.save(corrfile)
```

