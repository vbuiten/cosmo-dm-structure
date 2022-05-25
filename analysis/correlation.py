import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from matplotlib.ticker import AutoMinorLocator
from analysis.utils import *
from data.load import History

class CorrelationFunction:
    '''
    Class for estimating and plotting correlation functions from simulation data. The Landy-Szalay estimator is used
    for estimating the pair correlation function from the particle positions.

    Attributes:
        history: History instance
            History object containing the simulation data.
        positions_final: ndarray of shape (n_particles, dim)
            Positions of the particles at the end of the simulation.
        scale_factor_final: float
            Scale factor of the universe at the end of the simulation.
        distance_mids: ndarray of shape (n_bins,)
            Midpoints of the distance bins in which the correlation function is evaluated.
        corr_func: ndarray of shape (n_bins,)
            Correlation function evaluated in each distance bin specified by distance_mids.
        fig: matplotlib Figure instance
            Figure containing the correlation function plot.
        ax: matplotlib Axes instance
            Axes object containing the correlation function plot.
        label: str
            Label indicating the specific cosmological model used.

        Methods:
            plot():
                Plot the correlation function.
            addOther(other_corr_func):
                Add another CorrelationFunction object's correlation function to the plot.
            show()
                Show the figure.
            save(savefile):
                Save the figure.
    '''

    def __init__(self, history, logbins=False):
        '''

        Args:
            history: History instance
                History object containing the simulation data.
            logbins: bool
                If True, logarithmically-spaced distance bins are used. Default is False.
        '''

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
        '''
        Plot the figure.

        Returns:

        '''

        self.ax.plot(self.distance_mids, self.corr_func, label=self.label, alpha=0.7, lw=2)
        self.ax.legend()


    def addOther(self, other_corr_func):
        '''
        Add another CorrelationFunction instance's correlation function to the plot.

        Args:
            other_corr_func: CorrelationFunction instance
                The CorrelationFunction to add to the figure.
        Returns:

        '''

        self.ax.plot(other_corr_func.distance_mids, other_corr_func.corr_func,
                     label=other_corr_func.label, alpha=0.7, lw=2)
        self.ax.legend()


    def show(self):
        '''
        Show the figure.

        Returns:

        '''

        self.fig.show()


    def save(self, savefile):
        '''
        Save the figure.

        Args:
            savefile: str
                File to save the figure to.

        Returns:

        '''

        self.fig.savefig(savefile)