import numpy as np
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.utils import *

class ParticleMesh:
    def __init__(self, grid, particles, method="CIC"):

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

        if method == "NGP" or method == "CIC":
            self.method = method
        else:
            raise ValueError("Method must be 'NGP' or 'CIC'.")


    def densityFromParticles(self):

        if self.method == "NGP":
            # use the Nearest Grid Point method for now
            self.grid.densities = nearestGridPointDensity(self.particles.positions,
                                                            self.grid.mids_tuple)

        elif self.method == "CIC":
            self.grid.densities = cloudInCellDensity(self.particles.positions, self.grid.mids_tuple,
                                                     self.size)