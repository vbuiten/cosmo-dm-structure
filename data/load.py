import numpy as np
import h5py

class History:
    def __init__(self, savefile):

        if isinstance(savefile):
            if not savefile.endswith(".hdf5"):
                savefile = savefile + ".hdf5"
        else:
            raise TypeError("Argument 'savefile' must be a string.")

        dfile = h5py.File(savefile, "r")

        dset_positions = dfile["positions"]
        dset_scale_factor = dfile["scale-factors"]

        self.positions = np.copy(dset_positions)
        self.scale_factors = np.copy(dset_scale_factor)
        self.size = dfile.attrs["linear-size"]
        self.cell_size = dfile.attrs["cell-size"]

        self.n_times, self.n_particles, self.dim = self.positions.shape

        dfile.close()