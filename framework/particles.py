import numpy as np

class ParticleSet:
    '''
    Class for keeping track of the particles in a simulation.

    Attributes:
        positions: ndarray of shape (n_particles, dim)
            Comoving coordinates of the particles.
        momenta: ndarray of shape (n_particles, dim)
            "Momenta" of the particles. Momentum here is defined as the peculiar velocity divided by the scale factor.
    '''

    def __init__(self, size, dim, n_particles):

        if isinstance(size, int):
            self.size = size
        else:
            raise TypeError("size must be an integer.")

        if isinstance(dim ,int) and (dim == 2 or dim == 3):
            self.dim = dim
        else:
            raise ValueError("dim must be an integer with value 2 or 3.")

        if isinstance(n_particles, int) and n_particles >= 0:
            self.n_particles = n_particles
        else:
            raise TypeError("n_particles must be a non-negative integer.")


    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, comoving_coords):

        if isinstance(comoving_coords, np.ndarray) and comoving_coords.shape == (self.n_particles, self.dim):
            self._positions = comoving_coords

        else:
            raise TypeError("Argument must be of shape (n_particles, dim).")


    @property
    def momenta(self):
        return self._momenta

    @momenta.setter
    def momenta(self, mom):

        if isinstance(mom, np.ndarray) and mom.shape == (self.n_particles, self.dim):
            self._momenta = mom

        else:
            raise TypeError("Argument must be of shape (n_particles, dim).")