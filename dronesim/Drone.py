import numpy as np
from numpy.linalg import norm
from .quaternion_helpers import angle_between, quat_apply, quat_inv, quat_mult, quat_from_R, unit, cross_matrix
from .algorithms import EKF
from typing import Callable
import pandas as pd

class Drone:

    def __init__(self, dt: float, state0: np.ndarray):

        if dt <= 0:
            raise ValueError("dt must be greater than 0")

        self.state = state0.copy()
        self.fsm_state = "idle"
        self.full_navigation = False
        self.dt = dt
        self.t = 0

        # Must initialize everything with functions
        self.mass = None
        self.F_g = None
        self.I_inv = None
        self.dimensions = None
        self.dead = False

        # Erorrs
        self.prev_angle_error = 0  # rad
        self.prev_p_error = 0
        self.q_err_prev = None

        # Functions
        self.get_sim_state = None
        self.get_sim_time = None
        self.get_navigation_data = None

        # arrays for debugging
        self.a_body_array: list[np.ndarray] = []
        self.w_body_array: list[np.ndarray] = []
        self.p_glob_array: list[np.ndarray] = []

    def add_sim_functions(
        self,
        sim_state_func: Callable[[], np.ndarray],
        sim_time_func: Callable[[], float],
    ):
        self.get_sim_state = sim_state_func
        self.get_sim_time = sim_time_func

    def add_navigation_data_functions(
        self,
        a_w_p_data_func: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]],
        full_navigation = True
    ):
        self.get_navigation_data = a_w_p_data_func
        self.full_navigation = full_navigation
           
    def make_ekf(self,
        P0: np.ndarray,
        accel_bias: np.ndarray,
        gyro_bias: np.ndarray,
        lidar_bias: np.ndarray
        ):
        self.ekf = EKF(self.state[0:10], P0, self.dt)
        self.ekf.add_biases(accel_bias, gyro_bias, lidar_bias)

    def add_path(self, path_arr: dict[float, list[float]]):
        self.path = path_arr.copy()

    def add_dataframe_path(self, path_arr: pd.DataFrame):
        self.path = path_arr.copy()



    ############################################################################################################
    #                                        Drone Initialization                                              #
    ############################################################################################################

    def define_prop(
        self,
        arm_distance: float,
        prop_height: float,
        max_force_kgf: float,
        min_force_kgf: float,
        num: int = 4,
        kd: float = 0.01,
    ):

        if arm_distance < 0:
            raise ValueError("Arm distance must be positive")

        self.arm_distance = arm_distance
        self.prop_height = prop_height

        self.num_prop = num
        self.kd = kd

        self.force_bounds_N = [min_force_kgf * 9.81, max_force_kgf * 9.81]
        self.max_thrust_N = 4 * max_force_kgf * 9.81
        self.min_thrust_N = 4 * min_force_kgf * 9.81

        # Allocation Matrix
        # Reference: https://www.cantorsparadise.org/how-control-allocation-for-multirotor-systems-works-f87aff1794a2/
        r = arm_distance / np.sqrt(2)  # Distance from prop to central axis

        min_torque = 2 * r * self.min_thrust_N
        max_torque = 2 * r * self.max_thrust_N
        self.max_torque_X_Y_Nm = max_torque - min_torque
        
        self.max_torque_Z_Nm = 2 * kd * r * (max_force_kgf - min_force_kgf) * 9.81


        # SEEMED TO WORK FOR THE CONTROLLER
        allocation_matrix = np.array(
            [
                [1, 1, 1, 1],
                [r, r, -r, -r],
                [-r, r, r, -r],
                [kd, -kd, kd, -kd],
            ]
        )

        self.A = allocation_matrix
        self.A_inv = np.linalg.inv(allocation_matrix)

    def define_drone(self, mass: float, I: np.ndarray[float], dimensions: list[float]):  # noqa: E741

        if mass <= 0:
            raise ValueError("Mass must be greater than 0 kg")

        if np.shape(I) != (3, 3):
            raise ValueError("I matrix must be 3x3")

        if len(dimensions) != (3):
            raise ValueError("dimensions must be a 3-element list with X,Y,Z lengths")

        if type(dimensions) is list:
            self.dimensions = np.array([dimensions])
        else:
            self.dimensions = dimensions

        assert np.shape(dimensions) == (3,)

        self.mass = mass
        self.F_g = 9.81 * self.mass
        self.I = I.copy()
        self.I_inv = np.linalg.inv(I)

    ############################################################################################################
    #                                         GNC Initialization                                               #
    ############################################################################################################

    def set_attitude_controller(self, Kp: np.ndarray, Kd: np.ndarray, Lambda: np.ndarray = np.zeros((3,3))):
        assert np.shape(Kp) == (3, 3)
        assert np.shape(Kd) == (3, 3)
        assert np.shape(Lambda) == (3, 3)

        self.attitude_controller_Kp = Kp.copy()
        self.attitude_controller_Kd = Kd.copy()
        self.attitude_controller_2_Lambda = Lambda.copy()

    def set_position_controller(self, Kp: np.ndarray, Kd: np.ndarray):
        assert np.shape(Kp) == (3, 3)
        assert np.shape(Kd) == (3, 3)

        self.position_controller_Kp = Kp.copy()
        self.position_controller_Kd = Kd.copy()

    ############################################################################################################
    #                                                  Loop                                                    #
    ############################################################################################################

    ##########################################################################
    #                                Controls                                #
    ##########################################################################
    """
        - Create desired force/torque with controllers
        - Allocates desired force/torque to propellers
    """

    def position_controller_1(self, p_desired_L: np.ndarray, v_desired_L: np.ndarray, vertical_angle: float):
        """Broken as kinda hell"""
        assert np.shape(p_desired_L) == (3,)
        assert np.shape(v_desired_L) == (3,)


        p = self.p_calc
        v = self.v_calc

        kp = self.position_controller_Kp
        kd = self.position_controller_Kd


        p_err = p_desired_L - p
        v_err = v_desired_L - v

        # Force
        self.F_desired = np.matmul(kp,p_err.T) + np.matmul(kd,v_err.T) + np.array([0,0,self.F_g]).T

        # Clip to maximum force
        if norm(self.F_desired) > self.max_thrust_N:
            self.F_desired = self.F_desired * abs(self.max_thrust_N / norm(self.F_desired))

        # Thrust scaling -> https://www.desmos.com/calculator/gsl7czi1f2
        if self.F_desired[2] < 0: # TODO: make better condition for this? seems to work very well
            self.F_desired[2] = (self.F_g - self.min_thrust_N) * ( np.arctan(self.F_desired[2] / (self.F_g - self.min_thrust_N) * np.pi/2) + np.pi/2 ) * 2 / np.pi + self.min_thrust_N
        # else: # TODO: maybe use????
        #     self.F_desired[2] = (self.max_thrust_N - self.F_g) * ( np.arctan(self.F_desired[2] / (self.max_thrust_N - self.F_g) * np.pi/2) ) * 2 / np.pi  + self.F_g
        
        # Construct orthogonal frame to find desired quaternion
        z_axis_hat = unit(self.F_desired)
        x_axis_hat = unit(np.cross(z_axis_hat, np.cross(np.array([1,0,0]), z_axis_hat)) ) # assigns heading based off of X axis
        y_axis_hat = unit(np.cross(z_axis_hat, x_axis_hat))

        R = np.column_stack((x_axis_hat, y_axis_hat, z_axis_hat))
        q_des = unit(quat_from_R(R))

        thrust = norm(self.F_desired)

        # breakpoint()

        return q_des, thrust

    def position_controller_2(self, p_desired_L: np.ndarray,
                              v_desired_L: np.ndarray, a_desired_L: np.ndarray,
                              w_desired_L: np.ndarray, n_desired_L: np.ndarray,
                              theta_d: float):
        assert np.shape(p_desired_L) == (3,)
        assert np.shape(v_desired_L) == (3,)

        p = self.p_calc
        v = self.v_calc
        # w = self.w_calc

        kp = self.position_controller_Kp
        kd = self.position_controller_Kd

        w_desired_L = quat_apply(self.q_calc, w_desired_L)
        n_desired_L = quat_apply(self.q_calc, n_desired_L)

        # R_d = np.eye(3) + np.sin(theta_d) * cross_matrix(n_desired_L) + (1 - np.cos(theta_d)) * (cross_matrix(n_desired_L) @ cross_matrix(n_desired_L))
        # q_B2L = quat_from_R(R_d)
        # n_B = quat_apply(quat_inv(q_B2L), [0,0,1])
        # print(self.t, end="")
        # print(n_B)

        p_err = p - p_desired_L
        v_err = v - v_desired_L



        a_tot = a_desired_L - kp @ p_err - kd @ v_err + np.array([0,0,9.81])
        
        e3 = np.array([0,0,1])
        a_hat = unit(a_tot)
        thrust = self.mass * norm(a_tot)

        # print(up.T @ a_hat)
        # print(np.cross(up.T,a_hat))

        if norm(a_tot) < 0.0001:
            q_d = np.array([1,0,0,0])
        else:

            term1 = 1 / np.sqrt(2 * (1 + np.dot(e3, a_hat) ))
            term2 = np.array([1 + e3.T @ a_hat, *list(np.cross(e3, a_hat))])

            q_d = term1 * term2

            q_d = unit(q_d)


        dot = np.dot(e3, a_hat)
        axis = np.cross(e3, a_hat)
        if np.linalg.norm(axis) < 1e-6:
            q_d = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot)
            q_d = np.hstack((np.cos(angle/2), axis*np.sin(angle/2)))
        # omega_d = np.zeros(3)
        # return T_d, q_d, omega_d

        self.F_desired = 0

        # print(f"{p_desired_L} {p_err} {v_err} {a_tot}")



        return q_d, thrust


    def attitude_controller_1(
        self, q_desired_L: np.ndarray, w_desired_L: np.ndarray
    ) -> np.ndarray:
        assert np.shape(q_desired_L) == (4,)
        assert np.shape(w_desired_L) == (3,)

        kp = self.attitude_controller_Kp
        kd = self.attitude_controller_Kd

        q_error_L = quat_mult(quat_inv(q_desired_L), self.q_calc)
        w_error_L = self.w_calc - w_desired_L

        torque_L = -q_error_L[0] * kp @ q_error_L[1:4] - kd @ w_error_L.T


        # Clip torques based on max, but I'm not sure this is even being used
        # torque_L[0:2] = 2 * self.max_torque_X_Y_Nm * np.arctan(torque_L[0:2] * np.pi / 2 / self.max_torque_X_Y_Nm) / np.pi
        # torque_L[2] = 2 * self.max_torque_Z_Nm * np.arctan(torque_L[2] * np.pi / 2 / self.max_torque_Z_Nm) / np.pi

        return torque_L


    def attitude_controller_2(
        self, q_desired_L: np.ndarray, w_desired_L: np.ndarray
    ) -> np.ndarray:
        assert np.shape(q_desired_L) == (4,)
        assert np.shape(w_desired_L) == (3,)

        kp = self.attitude_controller_Kp
        kd = self.attitude_controller_Kd
        Lambda = self.attitude_controller_2_Lambda


        q_error_L = quat_mult(quat_inv(q_desired_L), self.q_calc)
        w_error_L = self.w_calc - w_desired_L
        q_err_dot = np.zeros((3,)) if self.q_err_prev is None else (q_error_L[1:4] - self.q_err_prev[1:4]) / self.dt
        # breakpoint()
        self.q_err_prev = q_error_L
        torque_L = -np.sign(q_error_L[0]) * kp @ q_error_L[1:4] - kd @ w_error_L - q_err_dot @ Lambda * np.sign(q_error_L[0])


        # Clip torques based on max, but I'm not sure this is even being used
        # torque_L[0:2] = 2 * self.max_torque_X_Y_Nm * np.arctan(torque_L[0:2] * np.pi / 2 / self.max_torque_X_Y_Nm) / np.pi
        # torque_L[2] = 2 * self.max_torque_Z_Nm * np.arctan(torque_L[2] * np.pi / 2 / self.max_torque_Z_Nm) / np.pi

        return torque_L


    def allocate_thrusts(self, thrust_z_B: float, torques_B: np.ndarray) -> np.ndarray:
        # Reference:
        # https://www.cantorsparadise.org/how-control-allocation-for-multirotor-systems-works-f87aff1794a2/

        outputs = np.concatenate(([thrust_z_B], torques_B))
        inputs = np.matmul(self.A_inv, outputs)

        return inputs

    def apply_motor_bounds(self, commands: np.ndarray):

        result = np.clip(
            commands, a_min=self.force_bounds_N[0], a_max=self.force_bounds_N[1]
        )
        return result

    ############################################################################################################
    #                                                Running                                                   #
    ############################################################################################################

    def state_machine(self):

        match self.fsm_state:

            case "hover_calibration":
                # Hovers at a specific position with no actions

                pass

            case "load_trajectory":
                # Hovers at a specific position with path planning
                # Sends trajectory to operator
                """For loop going through different time horizons until no bounding"""
                pass

            case "await_confirmation":
                # Idle, but awaiting confirmation from operator
                pass

            case "fly":
                # 💃
                pass

            case _:
                pass
    
    def get_position_desired(self):

        if type(self.path) is dict:

            timestamps = self.path.keys()

            key = max(i for i in timestamps if i < self.t)

            row = self.path[key]
            p_d = row[0]
            v_d = row[1]

            return np.array(p_d), np.array(v_d), np.array(np.zeros(3)), np.array(np.zeros(3)), np.array(np.zeros(3)), 0
        
        elif type(self.path) is pd.DataFrame:

            timestamps = self.path["t"]
            p_arr = self.path[["r_x", "r_y", "r_z"]]
            v_arr = self.path[["v_x", "v_y", "v_z"]]
            a_arr = self.path[["a_x", "a_y", "a_z"]]
            n_arr = self.path[["n_x", "n_y", "n_z"]]
            w_arr = self.path[["omega_x", "omega_y", "omega_z"]]
            theta_arr = self.path["theta"]
            
            # breakpoint()
            # row = max(timestamps.index[i] for i in range(len(timestamps)) if timestamps[i] < self.t)
            row = sum([1 for i in timestamps if i < self.t]) - 1
            
            # print(row)
            # breakpoint()

            p_d = p_arr.iloc[row]
            v_d = v_arr.iloc[row]
            a_d = a_arr.iloc[row]
            n_d = n_arr.iloc[row]
            w_d = w_arr.iloc[row]
            theta_d = theta_arr.iloc[row]
            # breakpoint()

            return np.array(p_d), np.array(v_d), np.array(a_d), np.array(w_d), np.array(n_d), float(theta_d)
    
        
        else:
            raise ValueError(f"Type of self.path is {type(self.path)}")

    def timestep(self):

        ####### State Machine ########
        """Governs which controller/trajectory to follow"""

        ######### Navigation #########

        self.t += self.dt
        sim_state = self.get_sim_state()
        self.a_meas, self.w_meas, self.p_meas = self.get_navigation_data()
        self.a_body_array.append(self.a_meas)
        self.w_body_array.append(self.w_meas)
        self.p_glob_array.append(self.p_meas)


        if self.full_navigation:

            self.ekf.predict(self.a_meas, self.w_meas)
            # print(ekf.state)

            # if random.random() < 0.1:
            self.ekf.update(self.p_meas)

            # Calculated state is actual state
            # self.p_calc = sim_state[0:3]
            # self.v_calc = sim_state[3:6]
            # self.q_calc = sim_state[6:10]
            # self.w_calc = sim_state[10:13]

            # # KALMAN
            self.p_calc = self.ekf.state[0:3]
            self.v_calc = self.ekf.state[3:6]
            self.q_calc = self.ekf.state[6:10]
            self.w_calc = self.w_meas
            
                        
        else:
                        

            # Calculated state is actual state
            self.p_calc = sim_state[0:3]
            self.v_calc = sim_state[3:6]
            self.q_calc = sim_state[6:10]
            self.w_calc = sim_state[10:13]


        # FIltering
        # a,w -> p,v,q,w

        self.state = np.concat([self.p_calc, self.v_calc, self.q_calc, self.w_calc])

        ######### Guidance #########

        ######### Control #########

        self.motor_forces = np.zeros(4)

        # q_d = np.array([1., 0., 0., 0.])
        # q_d = quat_from_axis_rot(10, [0, 1, 0])

        
        # w_d = np.array([0, 0, 100]) * DEG2RAD

        
        
        p_d, v_d, a_d, w_d, n_d, theta_d = self.get_position_desired()

        self.p_d_err = p_d - self.p_calc

        # v_d = np.zeros(3)

        vertical_axis = quat_apply(self.q_calc, [0, 0, 1])
        vertical_angle = angle_between(vertical_axis, [0, 0, 1])

        # print(vertical_angle)

        q_d, thrust = self.position_controller_2(p_d, v_d, a_d, w_d, n_d, theta_d)
        # q_d, thrust = self.position_controller_1(p_d, v_d, vertical_angle)
        # breakpoint()
        # q_d = np.array([1., 0., 0., 0.])
        # w_d = np.zeros(3)
        # thrust = self.F_g / np.cos(vertical_angle)
        torques = self.attitude_controller_2(q_d, w_d)


        # stepping = False
        # if (abs(torques[2]) > 0.0000001
        #     or stepping):
        #     # stepping = True
        #     breakpoint()

        # thrust = np.clip(thrust, a_min=self.min_thrust_N, a_max=self.max_thrust_N)

        # if vertical_angle * 180 / np.pi >= 89:
        #     self.dead = True

        thrust = thrust


        # breakpoint()
        self.motor_forces += self.apply_motor_bounds(
            self.allocate_thrusts(thrust, torques)
        )

        # self.motor_forces += self.allocate_thrusts(thrust, torques)


        self.torques = torques
        self.thrust = thrust
        self.vertical_angle = vertical_angle
        self.q_d = q_d


        ######### Propogation? #########



# TODO: SPIN CONTROL!?!?! Aborts current command to go vertical and nullify vertical velocity