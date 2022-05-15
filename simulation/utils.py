import numpy as np

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