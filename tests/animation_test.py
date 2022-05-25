'''Test file for testing the code for creating an animation.'''

from data.animation import Animation3D

datafolder = "/net/vdesk/data2/buiten/COP/cosmo_sims_data/"
datafile = datafolder + "size10_N1000_test.hdf5"
savefile = datafolder + "size10_N1000_test_animation.mp4"

animation = Animation3D(datafile)
animation.animate()
animation.save(savefile)