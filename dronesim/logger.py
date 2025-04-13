from .drone import Drone
from .simulation import Simulation
import numpy as np
from .quaternion_helpers import *


class Logger:

    def __init__(self, t_max: float, dt: float):

        self.drone = None

        steps = int(t_max / dt) + 1

        # State
        self.actual_states = np.zeros((steps, 13))
        self.calculated_states = np.zeros((steps, 13))

        # Force/Torque
        self.actual_forces = np.zeros((steps, 3))
        self.actual_torques = np.zeros((steps, 3))
        self.calculated_forces = np.zeros((steps, 3))
        self.calculated_torques = np.zeros((steps, 3))

        # Time
        self.t = np.linspace(0, t_max, steps)

    def add_sim(self, sim: Simulation):
        self.sim = sim

    def add_drone(self, drone: Drone):
        self.drone = drone

    def log(self, step: int):

        if self.drone is None:
            raise RuntimeError("Add drone to log it, dummy")
        
        if self.sim is None:
            raise RuntimeError("Add sim to log it, dummy")
        
        # State
        self.actual_states[step, :] = self.sim.actual_state
        self.calculated_states[step, :] = self.drone.state

        self.actual_forces[step, :] = self.sim.total_force
        self.actual_torques[step, :] = self.sim.total_torque



        
        
