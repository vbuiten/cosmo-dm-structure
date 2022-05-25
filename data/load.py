import numpy as np
import h5py

class History:
    '''
    Class for loading the data of a pre-run simulation.

    Attributes:
        positions: ndarray of shape (n_times, n_particles, dim)
            Positions of the particles at every time step.
        scale_factors: ndarray of shape (n_times,)
            Scale factor of the simulated universe at every time step.
        size: int
            Dimensionless linear size of the box.
        cell_size: float
            Linear size of a grid cell (typically 1.).
        Om0: float
            Present-day matter density parameter.
        Ok0: float
            Present-day curvature density parameter.
        Ode0: float
            Present-day dark energy (cosmological constant) density parameter.

    '''

    def __init__(self, datafile):
        '''

        Args:
            datafile: str
                Relative/absolute name of the file from which to load the simulation data.
        '''

        if isinstance(datafile, str):
            if not datafile.endswith(".hdf5"):
                datafile = datafile + ".hdf5"
        else:
            raise TypeError("Argument 'savefile' must be a string.")

        dfile = h5py.File(datafile, "r")

        dset_positions = dfile["positions"]
        dset_scale_factor = dfile["scale-factors"]

        self.positions = np.copy(dset_positions)
        self.scale_factors = np.copy(dset_scale_factor)
        self.size = dfile.attrs["linear-size"]
        self.cell_size = dfile.attrs["cell-size"]

        # load cosmological parameters
        self.Om0 = dfile.attrs["Om0"]
        self.Ode0 = dfile.attrs["Ode0"]
        self.Ok0 = dfile.attrs["Ok0"]

        self.n_times, self.n_particles, self.dim = self.positions.shape

        dfile.close()