from framework.particle_mesh import ParticleMesh
from simulation.utils import *
import h5py

class Simulator:
    '''
    Class for evolving a system specified by a ParticleMesh object and certain cosmological parameters.

    Attributes:
        part_mesh: ParticleMesh instance
            ParticleMesh object containing information on the field and particles.
        timestep: float
            Time step to use in terms of the scale factor.
        scale_factor: float
            Scale factor in the current state of the simulation.
        Om0: float
            Present-day matter density parameter. Default is 0.3.
        Ok0: float
            Present-day curvature density parameter. Default is 0.
        Ode0: float
            Present-day dark energy (cosmological constant) density parameter. Default is 0.7.

    Methods:
        backwardsEulerHalfStep(scale_factor):
            Take half a step back in time to get momenta evaluated at half-steps. Used as a starting point for
            leapfrog integration.
        forwardsEulerHalfStep(scale_factor):
            Take half a step forward in time to get mmomenta evaluated at full steps again. To be used after integrating
            through a leapfrog scheme.
        step(scale_factor):
            Take a single simulation step.
        evolve(scale_end, savefile=None, save_step=0.001, save_err_thresh=1e-5):
            Evolve the simulation up to some desired scale factor. Automatically used backwards and forwards Euler
            before and after using leapfrog integration.
    '''

    def __init__(self, part_mesh, a_start, timestep, Om0=0.3, Ode0=0.7, Ok0=0.):
        '''

        :param part_mesh: ParticleMesh instance
            ParticleMesh object containing information on the field and particles.
        :param a_start: float
            Scale factor at which to start the simulation.
        :param timestep: float
            Time step to use in terms of the scale factor.
        :param Om0: float
            Present-day matter density parameter. Default is 0.3.
        :param Ode0: float
            Present-day dark energy (cosmological constant) density parameter. Default is 0.7.
        :param Ok0: float
            Present-day curvature density parameter. Default is 0.
        '''

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
        '''
        Take a half-step of backward Euler such that the momenta are now evaluated at half-times. This is useful for
        setting up a leapfrog integration scheme.

        :param scale_factor: float
            Scale factor in the current state of the simulation.
        :return:
        '''

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
        '''
        Take a half-step of forward Euler such that the momenta are evaluated at full steps again. This is useful to
        call after using leapfrog integration

        :param scale_factor: float
            Scale factor in the current state of the simulation.
        :return:
        '''

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
        '''
        Take a single step of the simulation using leapfrog integration. Assumes that the momenta are given at
        half-steps (i.e. that an initial backward Euler step has been used).

        :param scale_factor: float
            Scale factor in the current state of the simulation.
        :return:
        '''

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


    def evolve(self, scale_end, savefile=None, save_step=0.001, save_err_thresh=1e-5):
        '''
        Evolves the simulation and optionally saves the data. A leapfrog integration scheme is used; initial and final
        half-steps are included.

        :param scale_end: float
                Scale factor at which to end the simulation.
        :param savefile: NoneType or string
                File to save the data to. If None, the data is not stored. Default is None.
        :param save_step: float
                Time step (in terms of scale factor) after which to save the positions.
        :param save_err_thresh: float
                Maximum difference between the actual scale factor and the scale factor at which the positions are to
                be stored, expressed as a fraction of the saving time step.
        :return:
        '''

        scale_factors = np.arange(self.scale_factor, scale_end, self.timestep)
        self.backwardsEulerHalfStep(self.scale_factor)

        # store the particle positions at every save_step
        scale_factors_history = []
        positions_history = []

        for i, a in enumerate(scale_factors):

            if abs(((a - self.scale_factor) / save_step) % 1) < save_err_thresh:
                scale_factors_history.append(a)
                positions_history.append(self.part_mesh.particles.positions)

            self.step(a)

            if abs(((a - self.scale_factor) / save_step) % 1) < save_err_thresh:
                print ("Scale factor:", np.around(a,3))

        positions_history = np.array(positions_history)
        scale_factors_history = np.array(scale_factors_history)


        # save the simulation data to a file if savefile is not None
        if isinstance(savefile, str):
            if not savefile.endswith(".hdf5"):
                savefile = savefile + ".hdf5"

            file = h5py.File(savefile, "w")
            pos_dset = file.create_dataset("positions", data=positions_history)
            scale_fac_dset = file.create_dataset("scale-factors", data=scale_factors_history)
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