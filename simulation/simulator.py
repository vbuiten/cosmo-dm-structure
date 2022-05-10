import numpy as np
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from scipy.interpolate import RegularGridInterpolator
from IPython import embed

class Simulator:
    def __init__(self, part_mesh, a_start, timestep, Om0=0.3):

        if isinstance(part_mesh, ParticleMesh):
            self.part_mesh = part_mesh
        else:
            raise TypeError("'part_mesh' must be a ParticleMesh instance.")

        self.timestep = timestep
        self.scale_factor = a_start
        self.Om0 = 0.3


    def step(self):

        self.part_mesh.densityFromParticles()
        potential = self.part_mesh.grid.potential(self.Om0, self.scale_factor)



        # needs a factor f for cosmological effects
        # not yet using leapfrog
        new_pos = (self.part_mesh.particles.positions + self.part_mesh.particles.momenta \
                 * self.timestep / self.scale_factor**2) % self.part_mesh.size

        acceleration_grid = -np.array(np.gradient(potential.real)).T

        interpolator = RegularGridInterpolator([self.part_mesh.grid.x_mids[0,:], self.part_mesh.grid.y_mids[:,0]],
                                               acceleration_grid)
        acceleration_particles = interpolator(self.part_mesh.particles.positions)

        new_momenta = self.part_mesh.particles.momenta + acceleration_particles * self.timestep

        self.part_mesh.particles.positions = new_pos
        self.part_mesh.particles.momenta = new_momenta


    def evolve(self, scale_end):

        scale_factors = np.arange(self.scale_factor, scale_end, self.timestep)

        for a in scale_factors:
            self.step()

        print ("Evolution finished.")