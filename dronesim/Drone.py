import numpy as np
from .quaternion_helpers import *
from .filters import EKF
from typing import Callable
from pprint import pprint
from copy import copy
    

class Drone:

    def __init__(self, dt: float, state0: np.ndarray = None):

        if dt <= 0:
            raise ValueError("dt must be greater than 0")

        if state0 is not None and type(state0) is not np.ndarray:
            raise ValueError("If state0 is input, must be an ndarray")
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
        self.full_navigation = True
        
    
    def make_ekf(self,
        P0: np.ndarray,
        accel_bias: np.ndarray,
        gyro_bias: np.ndarray,
        lidar_bias: np.ndarray
        ):
        self.ekf = EKF(self.state[0:10], P0, self.dt)
        self.ekf.add_biases(accel_bias, gyro_bias, lidar_bias)

    def add_path(self, path_arr: dict[float, list[float]]):
        self.path_arr = path_arr.copy()


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
        kd: float = 0.02,
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
                [r, -r, r, -r],
                [-r, -r, r, r],
                [kd * r, -kd * r, -kd * r, kd * r],
            ]
        )

        self.A = allocation_matrix
        self.A_inv = np.linalg.inv(allocation_matrix)

    def define_drone(self, mass: float, I: np.ndarray[float], dimensions: list[float]):

        if mass <= 0:
            raise ValueError("Mass must be greater than 0 kg")

        if np.shape(I) != (3, 3):
            raise ValueError("I matrix must be 3x3")

        if len(dimensions) != (3):
            raise ValueError("dimensions must be a 3-element list with X,Y,Z lengths")

        self.mass = mass
        self.F_g = 9.81 * self.mass
        self.dimensions = np.array(dimensions)
        self.I = I.copy()
        self.I_inv = np.linalg.inv(I)

    ############################################################################################################
    #                                         GNC Initialization                                               #
    ############################################################################################################

    def set_attitude_controller_1(self, Kp: np.ndarray, Kd: np.ndarray):
        assert np.shape(Kp) == (3, 3)
        assert np.shape(Kd) == (3, 3)

        self.attitude_controller_1_Kp = Kp.copy()
        self.attitude_controller_1_Kd = Kd.copy()

    def set_position_controller_1(self, Kp: np.ndarray, Kd: np.ndarray):
        assert np.shape(Kp) == (3, 3)
        assert np.shape(Kd) == (3, 3)

        self.position_controller_1_Kp = Kp.copy()
        self.position_controller_1_Kd = Kd.copy()

    ############################################################################################################
    #                                                  Loop                                                    #
    ############################################################################################################



    ##########################################################################
    #                               Navigation                               #
    ##########################################################################
    """
        Obtaining current state.
        
        - Gets state OR noise data
        - Runs Kalman filter
    """
    ##########################################################################
    #                                Guidance                                #
    ##########################################################################
    """
        Where the path planning happens.

        - Generate errors based on flight path

    """

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

        max_angle = 80 * DEG2RAD


        p = self.p_calc
        v = self.v_calc
        q = self.q_calc

        kp = self.position_controller_1_Kp
        kd = self.position_controller_1_Kd


        p_err = p_desired_L - p
        v_err = v_desired_L - v

        # Force
        self.F_desired = np.matmul(kp,p_err.T) + np.matmul(kd,v_err.T) + np.array([0,0,self.F_g]).T

        # Clip to maximum force
        if norm(self.F_desired) > self.max_thrust_N * 1.1:
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

    def attitude_controller_1(
        self, q_desired_L: np.ndarray, w_desired_L: np.ndarray
    ) -> np.ndarray:
        assert np.shape(q_desired_L) == (4,)
        assert np.shape(w_desired_L) == (3,)

        kp = self.attitude_controller_1_Kp
        kd = self.attitude_controller_1_Kd

        q_error_L = quat_mult(quat_inv(q_desired_L), self.q_calc)
        w_error_L = self.w_calc - w_desired_L

        torque_L = -q_error_L[0] * np.matmul(kp, q_error_L[1:4].transpose()) - np.matmul(kd, w_error_L.transpose())

        # Clip torques based on max, but I'm not sure this is even being used
        torque_L[0:2] = 2 * self.max_torque_X_Y_Nm * np.arctan(torque_L[0:2] * np.pi / 2 / self.max_torque_X_Y_Nm) / np.pi
        torque_L[2] = 2 * self.max_torque_Z_Nm * np.arctan(torque_L[2] * np.pi / 2 / self.max_torque_Z_Nm) / np.pi

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

        timestamps = self.path_arr.keys()

        key = max(i for i in timestamps if i < self.t)

        row = self.path_arr[key]
        p_d = row[0]
        v_d = row[1]

        return np.array(p_d), np.array(v_d)


    def timestep(self):

        ####### State Machine ########
        """Governs which controller/trajectory to follow"""

        ######### Navigation #########

        self.t += self.dt
        sim_state = self.get_sim_state()

        if self.full_navigation:
            self.a_meas, self.w_meas, self.p_meas = self.get_navigation_data()

            # print(a_meas, w_meas, p_meas)
            self.a_body_array.append(self.a_meas)
            self.w_body_array.append(self.w_meas)
            self.p_glob_array.append(self.p_meas)

            self.ekf.predict(self.a_meas, self.w_meas)
            # print(ekf.state)

            self.ekf.update(self.p_meas)

            # Calculated state is actual state
            self.p_calc = sim_state[0:3]
            self.v_calc = sim_state[3:6]
            self.q_calc = sim_state[6:10]
            self.w_calc = sim_state[10:13]

            # # KALMAN
            # self.p_calc = self.ekf.state[0:3]
            # self.v_calc = self.ekf.state[3:6]
            # self.q_calc = self.ekf.state[6:10]
            # self.w_calc = self.w_meas
            
                        
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

        # q_d = np.array([0., 0., 0., 0.98901019])
        # q_d = quat_from_axis_rot(10, [0, 1, 0])

        w_d = np.zeros(3)
        # w_d = np.array([0, 0, 100]) * DEG2RAD

        
        
        p_d, v_d = self.get_position_desired()

        
            


        self.p_d_err = p_d - self.p_calc


        # v_d = np.zeros(3)

        vertical_axis = quat_apply(self.q_calc, [0, 0, 1])
        vertical_angle = angle_between(vertical_axis, [0, 0, 1])

        q_d, thrust = self.position_controller_1(p_d, v_d, vertical_angle)

        # Angular velocity check
        # if norm(self.w_calc) > 10:

        #     q_d = np.array([1,0,0,0])

        # print(q_d)
        # q_d = np.array([1,0,0,0])
        # thrust = self.vertical_sample_controller(vertical_angle)

        torques = self.attitude_controller_1(q_d, w_d)


        # stepping = False
        # if (abs(torques[2]) > 0.0000001
        #     or stepping):
        #     # stepping = True
        #     breakpoint()

        thrust = np.clip(thrust, a_min=self.min_thrust_N, a_max=self.max_thrust_N)

        # if vertical_angle * 180 / np.pi >= 89:
        #     self.dead = True

        thrust = thrust

        # print(torques)

        # breakpoint()


        self.motor_forces += self.apply_motor_bounds(
            self.allocate_thrusts(thrust, torques)
        )


        self.torques = torques
        self.thrust = thrust
        self.vertical_angle = vertical_angle
        self.q_d = q_d


        ######### Propogation? #########



# TODO: SPIN CONTROL!?!?! Aborts current command to go vertical and nullify vertical velocity