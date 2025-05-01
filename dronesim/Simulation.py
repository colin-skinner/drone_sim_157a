from .Drone import Drone
import numpy as np
from .quaternion_helpers import *
from .integrators import rk4_func, euler_func


class Simulation:
    """Define environment and adds forces to a drone"""

    def __init__(self, t_max: float, dt: float, state0: np.ndarray):

        self.dt = dt
        self.t = 0
        self.t_max = t_max
        self.actual_state = state0

        # List of actual forces and torques used to calculate net forces and torques at each timestep
        self.forces: list[np.ndarray] = []
        self.torques: list[np.ndarray] = []

    ########################################
    #      Drone Initialization            #
    ########################################

    def add_drone(self, drone: Drone):
        self.drone = drone

    def get_state(self):
        return self.actual_state

    def get_time(self):
        return self.t

    ########################################
    #        Forces and Torques            #
    ########################################

    def add_force(self, force: np.ndarray, r: np.ndarray):
        """Adds force in the global frame of the drone"""

        if np.shape(force) != (3,):
            raise ValueError("force must be a 3x1 vector")

        if np.shape(r) != (3,):
            raise ValueError("r_body must be a 3x1 vector")

        # print(f"{r=} {force=}")
        torque = np.cross(r, force)

        # print(f"{torque}")

        self.forces.append(force)
        self.torques.append(torque)

    def add_torque(self, torque: np.ndarray):
        """Adds force in the global frame of the drone"""

        if np.shape(torque) != (3,):
            raise ValueError("torque must be a 3x1 vector")
        self.torques.append(torque)

    def add_force_body(self, force_B: np.ndarray, r_B: np.ndarray):
        """Body version of adding a force"""
        q_B2L = self.actual_state[6:10]

        force = quat_apply(q_B2L, force_B)
        r = quat_apply(q_B2L, r_B)

        self.add_force(force, r)

    def add_torque_body(self, torque_B):
        """Body version of adding a force"""
        q_B2L = self.actual_state[6:10]

        torque = quat_apply(q_B2L, torque_B)

        self.add_torque(torque)

    def calc_forces(self):
        """Sums external forces and adds its own inputs"""
        self.total_force = sum(self.forces)  # I think this works
        self.total_torque = sum(self.torques)

    ########################################
    #           Simulation                 #
    ########################################

    def drone_state_deriv(self, t: float, state: np.ndarray):

        v = state[3:6]
        q_B2L = state[6:10]
        w = state[10:13]

        dpdt = v
        dvdt = self.total_force / self.drone.mass
        dqdt = 0.5 * quat_mult(q_B2L, [0, *w])
        dwdt = np.matmul(self.drone.I_inv, self.total_torque)

        term = -np.cross(w.transpose(), np.matmul(self.drone.I, w)) + self.total_torque
        # print(np.shape(term))
        # dwdt = np.matmul(self.drone.I_inv, term.transpose())
        # print(np.shape(dwdt))

        # print(dwdt)

        return np.array([*dpdt, *dvdt, *dqdt, *dwdt])

    def sim_props(self, motor_forces: np.ndarray, offsets: np.ndarray = np.empty):
        """TODO: add motor spinup/down delays"""
        arm_distance = self.drone.arm_distance
        prop_height = self.drone.prop_height

        front_left_r = arm_distance * np.array(unit([1, 1, prop_height]))
        front_right_r = arm_distance * np.array(unit([1, -1, prop_height]))
        back_left_r = arm_distance * np.array(unit([-1, 1, prop_height]))
        back_right_r = arm_distance * np.array(unit([-1, -1, prop_height]))

        front_left_F = np.array([0, 0, motor_forces[0]])
        front_right_F = np.array([0, 0, motor_forces[1]])
        back_left_F = np.array([0, 0, motor_forces[2]])
        back_right_F = np.array([0, 0, motor_forces[3]])

        self.add_force_body(front_left_F, front_left_r)
        self.add_force_body(front_right_F, front_right_r)
        self.add_force_body(back_left_F, back_left_r)
        self.add_force_body(back_right_F, back_right_r)

        # Z_axis torques TODO: figure out if this is right
        self.add_torque_body(np.array([0, 0, self.drone.kd * motor_forces[0]]))
        self.add_torque_body(np.array([0, 0, -self.drone.kd * motor_forces[1]]))
        self.add_torque_body(np.array([0, 0, -self.drone.kd * motor_forces[2]]))
        self.add_torque_body(np.array([0, 0, self.drone.kd * motor_forces[3]]))

    def sim_drone_timestep(self, gravity_en=True):
        """ Figure out order of when drone and sim have their state vectors bc now it is off by 1"""
        # self.actual_state = self.next_state
        self.t = self.t + self.dt

        if np.isclose(self.t % 10, 0, atol=0.01):
            print(f"Simulating t={int(self.t)}s")

        # Calculates drone motor forces based off of its state
        self.drone.timestep()
        assert self.drone.t == self.t

        # Gravity is the only external force?
        if gravity_en:
            gravity = np.array([0, 0, -9.81]) * self.drone.mass
            self.add_force(gravity, np.zeros(3))

        # Calculate how actuator inputs affect the forces/torques
        self.sim_props(
            self.drone.motor_forces
        )  # Adds motor forces to array based on drone dimensions

        # Sum forces
        self.calc_forces()

        new_state = euler_func(
            self.t, self.dt, self.actual_state, self.drone_state_deriv
        )

        # normalize quat
        new_state[6:10] = unit(new_state[6:10])

        # Resets variables before next iteration
        self.actual_state = new_state
        self.forces = []
        self.torques = []
