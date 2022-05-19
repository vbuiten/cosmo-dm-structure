from data.animation import Animation3D
from analysis.correlation import CorrelationFunction
import numpy as np

datafolder = "/net/vdesk/data2/buiten/COP/cosmo_sims_data/"
linear_size = 20
n_particles = 256
Om0s = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
corrs = []

for Om0 in Om0s:
    datafile = "{}flat_size{}_N{}_Omega_m_{}.hdf5".format(datafolder, linear_size, n_particles, Om0)
    animfile = "{}flat_size{}_N{}_Omega_m_{}.mp4".format(datafolder, linear_size, n_particles, Om0)

    animation = Animation3D(datafile)
    animation.animate()
    animation.save(animfile)

    print ("Finished animating for $\Omega_m = $ {}".format(Om0))

    corrs.append(CorrelationFunction(datafile))

# plot the correlation functions in one figure
corrs[0].plot()

for i in range(1, len(corrs)):
    #print (corrs[i].distance_mids)
    corrs[0].addOther(corrs[i])
corrs[0].show()
corrs[0].save("{}flat_size{}_N{}_corrfuncs.png".format(datafolder, linear_size, n_particles))