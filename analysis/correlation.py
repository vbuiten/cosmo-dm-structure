import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from matplotlib.ticker import AutoMinorLocator
from analysis.utils import *
from data.load import History

class CorrelationFunction:

    def __init__(self, history, logbins=False):

        if isinstance(history, History):
            self.history = history
        elif isinstance(history, str):
            self.history = History(history)
        else:
            raise TypeError("Argument 'history' must be either a History instance or a file name.")

        # get random positions
        random_positions = randomPositions(self.history.n_particles, self.history.dim, self.history.size)

        # get the latest positions
        self.positions_final = self.history.positions[-1]
        self.scale_factor_final = self.history.scale_factors[-1]

        # get all data-data pairs, random-random pairs and data-random pairs
        # and make a histogram of them

        if logbins:
            distance_edges = np.logspace(np.log10(0.0001), np.log10(self.history.size),
                                         int(np.sqrt(self.history.n_particles)))
        else:
            distance_edges = np.linspace(0., self.history.size/2, int(np.sqrt(self.history.n_particles)))

        counts_data, self.distance_mids = countPairs(self.positions_final, self.positions_final, distance_edges)
        counts_random, _ = countPairs(random_positions, random_positions, distance_edges)
        counts_data_random, _ = countPairs(self.positions_final, random_positions, distance_edges)

        # estimate the correlation function
        self.corr_func = corrFuncLandySzalay(counts_data, counts_random, counts_data_random)

        self.fig, self.ax = plt.subplots(dpi=240)
        self.fig.suptitle("Estimated Pair Correlation Function")
        self.ax.set_title(r"$a =$ {}".format(str(np.around(self.scale_factor_final, 3))))
        self.label = "$\Omega_m =$ {}; $\Omega_\Lambda =$ {}; $\Omega_k =$ {}".format(np.around(self.history.Om0, 3),
                                                                                       np.around(self.history.Ode0, 3),
                                                                                       np.around(self.history.Ok0, 3))

        self.ax.set_xlabel(r"Separation $r$")
        self.ax.set_ylabel(r"$\xi (r)$")
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.grid(which="major")
        self.ax.grid(which="minor", alpha=.1, color="grey")

        self.ax.set_xscale("log")


    def plot(self):

        self.ax.plot(self.distance_mids, self.corr_func, label=self.label, alpha=0.7, lw=2)
        self.ax.legend()


    def addOther(self, other_corr_func):

        self.ax.plot(other_corr_func.distance_mids, other_corr_func.corr_func,
                     label=other_corr_func.label, alpha=0.7, lw=2)
        self.ax.legend()


    def show(self):

        self.fig.show()


    def save(self, savefile):

        self.fig.savefig(savefile)