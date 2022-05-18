import numpy as np
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.utils import *
import h5py
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

        acceleration_particles = particlesAccelerationFromPotential(self.part_mesh.grid.mids_tuple, potential,
                                                                    self.part_mesh.particles.positions)

        f_factor = friedmannFactor(scale_factor, self.Om0, self.Ode0, self.Ok0)
        momenta_minus_half = self.part_mesh.particles.momenta - f_factor * acceleration_particles * (self.timestep / 2)
        self.part_mesh.particles.momenta = momenta_minus_half

        # now the particle momenta are saved at half-steps


    def forwardsEulerHalfStep(self, scale_factor):

        # compute the momenta at half-step n/2 + 1/2
        self.part_mesh.densityFromParticles()
        potential = self.part_mesh.grid.potential(self.Om0, scale_factor)

        acceleration_particles = particlesAccelerationFromPotential(self.part_mesh.grid.mids_tuple, potential,
                                                                    self.part_mesh.particles.positions)

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

        acceleration_particles = particlesAccelerationFromPotential(self.part_mesh.grid.mids_tuple, potential,
                                                                    self.part_mesh.particles.positions)

        new_mom = self.part_mesh.particles.momenta + f_factor_n * acceleration_particles * self.timestep
        new_pos = self.part_mesh.particles.positions + f_factor_plushalf * new_mom * self.timestep / half_scale_factor**2

        self.part_mesh.particles.positions = new_pos
        self.part_mesh.particles.momenta = new_mom


    def evolve(self, scale_end, savefile=None):
        '''
        Evolves the simulation and optionally saves the data.

        :param scale_end: float
                Scale factor at which to end the simulation.
        :param savefile: NoneType or string
                File to save the data to. If None, the data is not stored. Default is None.
        :return:
        '''

        scale_factors = np.arange(self.scale_factor, scale_end, self.timestep)
        self.backwardsEulerHalfStep(self.scale_factor)

        # store the particle positions at every point in time
        positions_history = np.zeros((len(scale_factors), self.part_mesh.n_particles, self.part_mesh.dim))

        for i, a in enumerate(scale_factors):

            positions_history[i] = self.part_mesh.particles.positions
            self.step(a)

            if abs((a - self.scale_factor) % 0.001) < 1e-5:
                print ("Scale factor:", np.around(a,3))


        # save the simulation data to a file if savefile is not None
        if isinstance(savefile, str):
            if not savefile.endswith(".hdf5"):
                savefile = savefile + ".hdf5"

            file = h5py.File(savefile, "w")
            pos_dset = file.create_dataset("positions", data=positions_history)
            scale_fac_dset = file.create_dataset("scale-factors", data=scale_factors)
            file.attrs["linear-size"] = self.part_mesh.size
            file.attrs["cell-size"] = 1.

            # save cosmological parameters
            file.attrs["Om0"] = self.Om0
            file.attrs["Ode0"] = self.Ode0
            file.attrs["Ok0"] = self.Ok0

            file.close()
            print ("File created at {}".format(savefile))


        self.forwardsEulerHalfStep(scale_end)
        self.scale_factor = scale_end

        print ("Evolution finished.")