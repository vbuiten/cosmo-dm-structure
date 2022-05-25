'''File for running and analysing an example 3D simulation. Also serves as an example for working with the code.'''

from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.simulator import Simulator
from data.animation import Animation3D
from analysis.correlation import CorrelationFunction

# set up the size of the box and number of particles
linear_size = 20
n_particles = 8**3
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
a_step = 1e-3
a_step_save = 1e-3
a_end = 3e-2

# specify the cosmological model
matter_dens = 0.3
lambda_dens = 0.7
curv_dens = 0.

# run and save the simulation
simfile = "size{}_N{}_Om0_{}_Ode0_{}_Ok0_{}.hdf5".format(linear_size, n_particles, matter_dens, lambda_dens, curv_dens)
sim = Simulator(pm, a_start, a_step, matter_dens, lambda_dens, curv_dens)
sim.evolve(a_end, simfile, save_step=a_step_save, save_err_thresh=1e-3)

print ("Finished the simulation.")

# create an animation
animfile = "size{}_N{}_Om0_{}_Ode0_{}_Ok0_{}.mp4".format(linear_size, n_particles, matter_dens, lambda_dens, curv_dens)
animation = Animation3D(simfile)
animation.animate()
animation.save(animfile)

# compute and plot the correlation function
corrfile = "size{}_N{}_Om0_{}_Ode0_{}_Ok0_{}_corr_func.hdf5".format(linear_size, n_particles, matter_dens, lambda_dens,
                                                                    curv_dens)
corr = CorrelationFunction(simfile)
corr.plot()
corr.show()
corr.save(corrfile)