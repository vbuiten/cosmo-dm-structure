import numpy as np
from scipy.interpolate import RegularGridInterpolator

def friedmannFactor(scale_factor, Om0=0.3, Ode0=0.7, Ok0=0.):
    '''
    Computes the "Friedmann factor" f(a), defined as the Hubble constant divided by the time derivative of the scale
    factor.
    
    :param scale_factor: float
            The scale factor a of the universe.
    :param Om0: float
            The present-day matter density parameter.
    :param Ode0: float
            The present-day dark energy (i.e. cosmological constant) density parameter.
    :param Ok0: float
            The present-day curvature density parameter.
    :return: f_factor: float
            The Friedmann factor f(a).
    '''

    densities_sum = Om0 + Ok0 * scale_factor + Ode0 * scale_factor**3
    f_factor = ( (1./scale_factor) * densities_sum)**(-1./2)

    return f_factor


def particlesAccelerationFromPotential(grid_mids_tuple, potential_field, particle_pos):
    '''
    Computes the acceleration at particle positions from the potential field evaluated on a grid.

    :param grid_mids_tuple: tuple or list or ndarray
            The grid cell midpoints in all dimensions, in the form of a tuple.
    :param potential_field: ndarray of shape (size, size, size) or (size, size)
            The field in the potential evaluated on a meshgrid (with Cartesian indexing).
    :param particle_pos: ndarray of shape (n_particles, size)
            The particle positions
    :return:
    '''

    acceleration_field = - np.array(np.gradient(np.flip(potential_field.real, axis=0))).T

    interpolator = RegularGridInterpolator(grid_mids_tuple, acceleration_field, bounds_error=False,
                                           fill_value=None)
    acceleration_particles = interpolator(particle_pos)

    return acceleration_particles