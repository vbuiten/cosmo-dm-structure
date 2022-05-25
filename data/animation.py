import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams["font.family"] = "serif"
from data.load import History



class Animation3D:
    '''
    Class for making 3D animations of the particle positions throughout a pre-run simulations.

    Attributes:
        history: History instance
            History object containing the simulation data.
        fig: matplotlib Figure instance
            Figure object for the simulation.
        ax: matplotlib Axes instance
            Axes object for the simulation.
        frame: matplotlib Line3D instance
            Object containing the drawn points in the plot.
        label_time: str
            Label indicating the scale factor at each time stamp.
        anim: matplotlib FuncAnimation instance
            The FuncAnimation object containing the animation.

    Methods:
        plotSnapshot(snapshot_idx):
        animate():
        save(savefile):
    '''

    def __init__(self, datafile):

        self.history = History(datafile)

        self.fig = plt.figure(dpi=240)
        self.ax = self.fig.add_subplot(projection="3d")

        self.frame, = self.ax.plot(self.history.positions[0,:,0], self.history.positions[0,:,1],
                                   self.history.positions[0,:,2],
                                   ls="", marker="o", alpha=.2, c="darkred")

        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")
        self.ax.set_zlabel(r"$z$")

        self.fig.suptitle(r"Simulation for $\Omega_m$ = {}; $\Omega_\Lambda$ = {};"
                          r" $\Omega_k$ = {}".format(self.history.Om0, self.history.Ode0, self.history.Ok0))

        self.label_time = self.ax.set_title("a = {}".format(np.around(self.history.scale_factors[0], 4)))


    def plotSnapshot(self, snapshot_idx):
        '''
        Plot one snapshot of the simulatoin data. Used as a helper function for creating the simulation.

        Args:
            snapshot_idx: int
                Index of the snapshot to plot.

        Returns:

        '''

        self.frame.set_xdata(self.history.positions[snapshot_idx,:,0])
        self.frame.set_ydata(self.history.positions[snapshot_idx,:,1])
        self.frame.set_3d_properties(self.history.positions[snapshot_idx,:,2])

        self.label_time.set_text("a = {}".format(np.around(self.history.scale_factors[snapshot_idx], 4)))

    def animate(self):
        '''
        Create the animation of the simulation data.

        Returns:

        '''

        self.anim = FuncAnimation(self.fig, self.plotSnapshot, frames=self.history.n_times,
                                  blit=False, cache_frame_data=False, repeat=False)

    def save(self, savefile):
        '''
        Save the created animation of the simulation.

        Args:
            savefile: str
                File to save the animation to (preferably mp4).

        Returns:

        '''

        writer = FFMpegWriter(fps=10)

        try:
            self.anim.save(savefile, writer=writer)
        except:
            plt.rcParams["animation.ffmpeg_path"] = r"C:\Program Files\FFmpeg\bin\ffmpeg.exe"
            self.anim.save(savefile, writer=writer)

        # save the figure to avoid overplotting
        self.fig.clf()

        print ("Saved animation.")