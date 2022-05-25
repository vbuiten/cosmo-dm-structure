'''Test file for testing the code for computing and plotting correlation functions.'''

from analysis.correlation import CorrelationFunction

datafolder = "/net/vdesk/data2/buiten/COP/cosmo_sims_data/"
datafile = datafolder + "size10_N1000_test.hdf5"
savefile = datafolder + "size10_N1000_test_animation.mp4"

corr = CorrelationFunction(datafile)
corr.plot()
corr.show()