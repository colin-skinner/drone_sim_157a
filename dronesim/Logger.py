from .Drone import Drone
from .Simulation import Simulation
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
        self.step = 0

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

        # Commanded thrust and torques
        self.drone_commanded_thrust = np.zeros((steps, 1))
        self.drone_commanded_torques = np.zeros((steps, 3))
        self.drone_desired_quat = np.zeros((steps, 4))

        # Debug
        self.drone_p_d_error = np.zeros((steps, 3))


        # For Kalman filter training
        self.actual_a_body = np.zeros((steps, 3))
        self.actual_w_body = np.zeros((steps, 3))

        # Time
        self.t = np.linspace(0, t_max, steps)  # TODO: figure out if this works

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

        # Force/Torque
        self.actual_forces[step, :] = self.sim.total_force
        self.actual_torques[step, :] = self.sim.total_torque

        # Commanded thrust and torques
        self.drone_commanded_thrust[step] = self.drone.thrust
        self.drone_commanded_torques[step, :] = self.drone.torques
        self.drone_desired_quat[step, :] = self.drone.q_d

        # DEBUG
        self.drone_p_d_error[step, :] = self.drone.p_d_err

        # For Kalman filter training
        q_L2B = quat_inv(self.drone.state[6:10])
        w_body = self.sim.actual_state[10:13]
        a_body = self.sim.total_force / self.drone.mass


        self.actual_a_body[step, :] = quat_apply(q_L2B, a_body)
        self.actual_w_body[step, :] = quat_apply(q_L2B, w_body)
        self.step = step

    def create_dataframe(self):
        self.results = pd.DataFrame()

        # Time
        self.results["Time (s)"] = self.t

        # Actual states
        self.results[
            [
                "x_actual (m)",
                "y_actual (m)",
                "z_actual (m)",
                "vx_actual (m/s)",
                "vy_actual (m/s)",
                "vz_actual (m/s)",
                "qw_actual",
                "qx_actual",
                "qy_actual",
                "qz_actual",
                "wx_actual (rad/s)",
                "wy_actual (rad/s)",
                "wz_actual (rad/s)",
            ]
        ] = self.actual_states

        # Drone states
        self.results[
            [
                "x_drone (m)",
                "y_drone (m)",
                "z_drone (m)",
                "vx_drone (m/s)",
                "vy_drone (m/s)",
                "vz_drone (m/s)",
                "qw_drone",
                "qx_drone",
                "qy_drone",
                "qz_drone",
                "wx_drone (rad/s)",
                "wy_drone (rad/s)",
                "wz_drone (rad/s)",
            ]
        ] = self.drone_states

        # Actual Forces
        self.results[["Fx_actual (N)", "Fy_actual (N)", "Fz_actual (N)"]] = (
            self.actual_forces
        )
        # Actual Torques
        self.results[["Tx_actual (Nm)", "Ty_actual (Nm)", "Tz_actual (Nm)"]] = (
            self.actual_torques
        )

        self.results[["ax_body (m/s2)", "ay_body (m/s2)", "az_body (m/s2)"]] = (
            self.actual_a_body
        )

        self.results[["wx_body (rad/s2)", "wy_body (rad/s2)", "wz_body (rad/s2)"]] = (
            self.actual_w_body
        )

        # TODO: FIGURE OUT WHY THIS IS NOT WORKING WITH TEH CORRECT INDICES

    def save(self, filename: str = None):

        self.create_dataframe()

        if filename is None:
            time = datetime.now()
            filename = time.strftime("%Y_%m_%d-%H_%M_%S")

        filename = f"{os.getcwd()}/results/{filename}.csv"

        with open(filename, "w") as file:

            # file.write("WOW")
            file.write(self.results.to_csv(index=False))
