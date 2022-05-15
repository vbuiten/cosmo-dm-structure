import numpy as np
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from scipy.interpolate import RegularGridInterpolator
from simulation.utils import friedmannFactor
from IPython import embed

class Simulator:
    def __init__(self, part_mesh, a_start, timestep, Om0=0.3, Ode0=0.7, Ok0=0.):

        if isinstance(part_mesh, ParticleMesh):
            self.part_mesh = part_mesh
        else:
            raise TypeError("'part_mesh' must be a ParticleMesh instance.")

        self.timestep = timestep
        self.scale_factor = a_start
        self.Om0 = Om0
        self.Ok0 = Ok0
        self.Ode0 = Ode0


    def backwardsEulerHalfStep(self, scale_factor):

        # compute the momenta at half-step n = -1/2
        self.part_mesh.densityFromParticles()
        potential = self.part_mesh.grid.potential(self.Om0, scale_factor)
        acceleration_grid = - np.array(np.gradient(potential.real)).T
        interpolator = RegularGridInterpolator([self.part_mesh.grid.mids1d, self.part_mesh.grid.mids1d],
                                               acceleration_grid, fill_value=0, bounds_error=False)
        acceleration_particles = interpolator(self.part_mesh.particles.positions)

        f_factor = friedmannFactor(scale_factor, self.Om0, self.Ode0, self.Ok0)
        momenta_minus_half = self.part_mesh.particles.momenta - f_factor * acceleration_particles * (self.timestep / 2)
        self.part_mesh.particles.momenta = momenta_minus_half

        # now the particle momenta are saved at half-steps


    def forwardsEulerHalfStep(self, scale_factor):

        # compute the momenta at half-step n/2 + 1/2
        self.part_mesh.densityFromParticles()
        potential = self.part_mesh.grid.potential(self.Om0, scale_factor)
        acceleration_grid = - np.array(np.gradient(potential.real)).T
        interpolator = RegularGridInterpolator([self.part_mesh.grid.mids1d, self.part_mesh.grid.mids1d],
                                               acceleration_grid, fill_value=0, bounds_error=False)
        acceleration_particles = interpolator(self.part_mesh.particles.positions)

        f_factor = friedmannFactor(scale_factor, self.Om0, self.Ode0, self.Ok0)

        momenta_plus_half = self.part_mesh.particles.momenta + f_factor * acceleration_particles * (self.timestep / 2)
        self.part_mesh.particles.momenta = momenta_plus_half

        # now the particle momenta are saved at whole steps again


    def step(self, scale_factor):

        self.part_mesh.densityFromParticles()
        potential = self.part_mesh.grid.potential(self.Om0, scale_factor)

        f_factor_n = friedmannFactor(scale_factor, self.Om0, self.Ode0, self.Ok0)
        half_scale_factor = scale_factor + self.timestep / 2
        f_factor_plushalf = friedmannFactor(half_scale_factor, self.Om0, self.Ode0, self.Ok0)

        acceleration_grid = - np.array(np.gradient(potential.real)).T
        interpolator = RegularGridInterpolator([self.part_mesh.grid.mids1d, self.part_mesh.grid.mids1d],
                                               acceleration_grid, fill_value=0., bounds_error=False)
        acceleration_particles = interpolator(self.part_mesh.particles.positions)

        new_mom = self.part_mesh.particles.momenta + f_factor_n * acceleration_particles * self.timestep
        new_pos = self.part_mesh.particles.positions + f_factor_plushalf * new_mom * self.timestep / half_scale_factor**2

        self.part_mesh.particles.positions = new_pos
        self.part_mesh.particles.momenta = new_mom


    def evolve(self, scale_end):

        scale_factors = np.arange(self.scale_factor, scale_end, self.timestep)
        self.backwardsEulerHalfStep(self.scale_factor)

        for a in scale_factors:
            self.step(a)

            if abs((a - self.scale_factor) % 0.001) < 1e-4:
                print ("Scale factor:", np.around(a,3))

        self.forwardsEulerHalfStep(scale_end)
        self.scale_factor = scale_end

        print ("Evolution finished.")