import numpy as np
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.utils import nearestGridPointDensity

class ParticleMesh:
    def __init__(self, grid, particles):

        if isinstance(grid, Grid):
            self.grid = grid
        else:
            raise TypeError("'grid' must be a Grid instance.")

        if isinstance(particles, ParticleSet):
            self.particles = particles
        else:
            raise TypeError("'particles' must be a ParticleSet instance.")

        self.dim = self.grid.dim
        self.n_particles = self.particles.n_particles
        self.size = self.grid.size


    def densityFromParticles(self):

        # use the Nearest Grid Point method for now
        if self.dim == 2:
            self.grid.densities = nearestGridPointDensity(self.particles.positions,
                                                          self.grid.mids1d)

        else:
            pass