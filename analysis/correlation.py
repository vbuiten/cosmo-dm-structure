import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from matplotlib.ticker import AutoMinorLocator
from analysis.utils import *
from data.load import History

class CorrelationFunction:

    def __init__(self, history):

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

        # get all data-data pairs, random-random pairs and data-random pairs
        # and make a histogram of them
        distance_edges = np.arange(0., self.history.size, 0.02*self.history.size)
        counts_data, self.distance_mids = countPairs(self.positions_final, self.positions_final, distance_edges)
        counts_random, _ = countPairs(random_positions, random_positions, distance_edges)
        counts_data_random, _ = countPairs(self.positions_final, random_positions, distance_edges)

        # estimate the correlation function
        self.corr_func = corrFuncLandySzalay(counts_data, counts_random, counts_data_random)

        self.fig, self.ax = plt.subplots(dpi=240)
        self.fig.suptitle("Estimated Pair Correlation Function")
        self.label = ("$\Omega_m =$ {}; $\Omega_\Lambda =$ {}; $\Omega_k =$ {}".format(self.history.Om0,
                                                                                       self.history.Ode0,
                                                                                       self.history.Ok0))

        self.ax.set_xlabel(r"Separation $r$")
        self.ax.set_ylabel(r"$\xi (r)$")
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.grid(which="major")
        self.ax.grid(which="minor", alpha=.1, color="grey")


    def plot(self):

        self.ax.plot(self.distance_mids, self.corr_func, label=self.label)
        self.ax.legend()


    def addOther(self, other_corr_func):

        self.ax.plot(other_corr_func.distance_mids, other_corr_func.corr_func,
                     label=other_corr_func.label)
        self.ax.legend()


    def show(self):

        self.fig.show()