import numpy as np
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.utils import *

class ParticleMesh:
    '''
    Class for handling the density assignment between the grid and the particles.

    Attributes:
        grid: Grid instance
            Grid object containing the information on the discretised field.
        particles: ParticleSet instance
            ParticleSet containing the information on the particles.
        dim: int
            Number of dimensions. Must be either 2 or 3.
        n_particles: int
            Number of particles.
        size: int
            Linear size of the box in terms of the number of grid cells.
        method: str
            Density assignment method to use. Valid inputs are "CIC" (cloud-in-cell) and "NGP" (nearest grid point).
            Default is "CIC".

    Methods:
        densityFromParticles():
            Computes the discretised density field from the particle positions, and assigns the density values to the
            grid cells.

    '''

    def __init__(self, grid, particles, method="CIC"):
        '''

        :param grid: Grid instance
            Grid object containing the information on the discretised field.
        :param particles: ParticleSet instance
            ParticleSet instance containing the information on the particles.
        :param method: str
            Density assignment method to use. Valid inputs are "CIC" (cloud-in-cell) and "NGP" (nearest grid point).
            Default is "CIC".
        '''

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
        '''
        Computes the discretised density field from the particle positions, and assigns the density values to the
        grid cells.

        :return:
        '''

        if self.method == "NGP":
            # use the Nearest Grid Point method for now
            self.grid.densities = nearestGridPointDensity(self.particles.positions,
                                                            self.grid.mids_tuple)

        elif self.method == "CIC":
            self.grid.densities = cloudInCellDensity(self.particles.positions, self.grid.mids_tuple,
                                                     self.size)