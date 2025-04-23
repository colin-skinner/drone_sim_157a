from .drone import Drone
from .simulation import Simulation
import numpy as np
from .quaternion_helpers import *
from datetime import datetime
import os
import pandas as pd

class Logger:

    def __init__(self, t_max: float, dt: float):

        self.drone = None
        self.sim = None
        self.results = None

        steps = int(t_max / dt) + 1

        # State
        self.actual_states = np.zeros((steps, 13))
        self.drone_states = np.zeros((steps, 13))
        self.drone_vertical_angle = np.zeros((steps, 1))

        # Force/Torque
        self.actual_forces = np.zeros((steps, 3))
        self.actual_torques = np.zeros((steps, 3))
        self.drone_forces = np.zeros((steps, 3))
        self.drone_torques = np.zeros((steps, 3))

        # Commadned thrust and torques
        self.drone_commanded_thrust = np.zeros((steps, 1))
        self.drone_commanded_torques = np.zeros((steps, 3))

        # Time
        self.t = np.linspace(0, t_max, steps) # TODO: figure out if this works

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
        self.drone_states[step, :] = self.drone.state
        self.drone_vertical_angle[step, :] = self.drone.vertical_angle

        self.actual_forces[step, :] = self.sim.total_force
        self.actual_torques[step, :] = self.sim.total_torque

        self.drone_commanded_thrust[step] = self.drone.thrust
        self.drone_commanded_torques[step, :] = self.drone.torques

    def create_dataframe(self):
        self.results = pd.DataFrame()

        # Time
        self.results["Time (s)"] = self.t

        # Actual states
        self.results[[
            "x_actual (m)",
            "y_actual (m)",
            "z_actual (m)",
            "vx_actual (m)",
            "vy_actual (m)",
            "vz_actual (m)",
            "qw_actual (m)",
            "qx_actual (m)",
            "qy_actual (m)",
            "qz_actual (m)",
            "wx_actual (m)",
            "wy_actual (m)",
            "wz_actual (m)"
            ]] = self.actual_states
        
        # Drone states
        self.results[[
            "x_drone (m)",
            "y_drone (m)",
            "z_drone (m)",
            "vx_drone (m)",
            "vy_drone (m)",
            "vz_drone (m)",
            "qw_drone (m)",
            "qx_drone (m)",
            "qy_drone (m)",
            "qz_drone (m)",
            "wx_drone (m)",
            "wy_drone (m)",
            "wz_drone (m)"
            ]] = self.drone_states
        
        # Actual Forces
        self.results[[
            "Fx_actual (N)",
            "Fy_actual (N)",
            "Fz_actual (N)"
            ]] = self.actual_forces
        # Actual Torques
        self.results[[
            "Tx_actual (N)",
            "Ty_actual (N)",
            "Tz_actual (N)"
            ]] = self.actual_torques

        # Drone Forces
        # self.results[[
        #     "Fx_drone (N)",
        #     "Fy_drone (N)",
        #     "Fz_drone (N)"
        #     ]] = self.drone_forces
        # Drone Torques
        
        

        
        


    def save(self, filename: str = None):

        self.create_dataframe()
        
        if filename is None:
            time = datetime.now()
            filename = time.strftime("%Y_%m_%d-%H_%M_%S")

        filename = f"{os.getcwd()}/results/{filename}.csv"
            
        with open(filename, 'w') as file:

            # file.write("WOW")
            file.write(self.results.to_csv(index=False))






        
        
