import numpy as np

class ParticleSet:
    '''
    Class for keeping track of the particles in a simulation.

    Attributes:
        positions: ndarray of shape (n_particles, dim)
            Comoving coordinates of the particles.
        momenta: ndarray of shape (n_particles, dim)
            "Momenta" of the particles. Momentum here is defined as the peculiar velocity divided by the scale factor.
        dim: int
            Number of dimensions. Must be either 2 or 3.
        n_particles: int
            Number of particles.
        size: int
            Linear size of the box in terms of the number of grid cells.

    Methods:
        uniformRandomPositions():
            Randomly draw particle positions from a uniform random distribution, such that the particles have equal
            probability to be anywhere in the box.
        zeroMomenta():
            Set the particle momenta to zero in all directions.
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
            # implement periodic boundary conditions such that the particles stay inside the box
            self._positions = comoving_coords % self.size

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


    def uniformRandomPositions(self):
        '''
        Draw random particle positions from a uniform distribution.

        :return:
        '''

        rng = np.random.default_rng()
        self.positions = rng.uniform(0, self.size, size=(self.n_particles, self.dim))


    def zeroMomenta(self):
        '''
        Set all particle momenta to zero.

        :return:
        '''

        self.momenta = np.zeros((self.n_particles, self.dim))