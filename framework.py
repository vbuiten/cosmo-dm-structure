import numpy as np

class ParticleGrid:
    '''
    Particle Mesh grid containing the density of each grid cell.

    '''

    def __init__(self, size):
        '''
        :param size: int
                Linear size of the box in terms of grid cells on each side
        '''

        mids = np.arange(0.5, size+0.5)
        self.x_mids, self.y_mids, self.z_mids = np.meshgrid(mids, mids, mids)

        # initialise an empty piece of universe
        self.densities = np.zeros_like(self.x_mids)