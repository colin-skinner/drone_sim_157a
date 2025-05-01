import numpy as np
from .quaternion_helpers import *
from typing import Callable
from pprint import pprint

class Drone:

    def __init__(self, dt: float, state0: np.ndarray = None):

        if dt <= 0:
            raise ValueError("dt must be greater than 0")

        if state0 is not None and type(state0) is not np.ndarray:
            raise ValueError("If state0 is input, must be an ndarray")
        self.state = state0
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

    def add_sim_functions(
        self,
        sim_state_func: Callable[[], np.ndarray],
        sim_time_func: Callable[[], float],
    ):
        self.get_sim_state = sim_state_func
        self.get_sim_time = sim_time_func

    def add_path(self, path_arr: dict[float, list[float]]):
        self.path_arr = path_arr

    ############################################################################################################
    #                                       Sensor Initialization                                              #
    ############################################################################################################

    def add_accel_noise(
        self,
        bias: np.ndarray = np.zeros(3),
        noise_std: np.ndarray = np.zeros(3),
    ):

        if len(bias) != 3:
            raise ValueError("Length of accelerometer bias array must be 3")

        if len(noise_std) != 3:
            raise ValueError(
                "Length of accelerometer standard deviation array must be 3"
            )

        self.accel_bias = np.array(bias)
        self.accel_noise_std = np.array(noise_std)

    def add_gyro_noise(
        self,
        bias: np.ndarray = np.zeros(3),
        noise_std: np.ndarray = np.zeros(3),
    ):

        if len(bias) != 3:
            raise ValueError("Length of gyroscope bias array must be 3")

        if len(noise_std) != 3:
            raise ValueError("Length of gyroscope standard deviation array must be 3")

        self.gyro_bias = np.array(bias)
        self.gyro_noise_std = np.array(noise_std)

    def add_lidar_noise(
        self,
        bias: np.ndarray = np.zeros(3),
        noise_std: np.ndarray = np.zeros(3),
    ):
        if len(bias) != 3:
            raise ValueError("Length of lidar bias array must be 3")

        if len(noise_std) != 3:
            raise ValueError("Length of lidar standard deviation array must be 3")

        self.lidar_bias = np.array(bias)
        self.lidar_noise_std = np.array(noise_std)

    def add_imu_misalignment(self, m_prime_to_m: np.ndarray):
        assert len(m_prime_to_m) == 4
        m_prime_to_m = unit(m_prime_to_m)

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

        self.force_bounds_N = [min_force_kgf * 9.81, max_force_kgf * 9.81]
        self.max_thrust_N = 4 * max_force_kgf * 9.81
        self.min_thrust_N = 4 * min_force_kgf * 9.81
        self.num_prop = num
        self.kd = kd

        # Allocation Matrix
        # Reference: https://www.cantorsparadise.org/how-control-allocation-for-multirotor-systems-works-f87aff1794a2/
        r = arm_distance / np.sqrt(2)  # Distance from prop to central axis

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
        self.I = I
        self.I_inv = np.linalg.inv(I)

    ############################################################################################################
    #                                         GNC Initialization                                               #
    ############################################################################################################

    def set_attitude_controller_1(self, Kp: np.ndarray, Kd: np.ndarray):
        assert np.shape(Kp) == (3, 3)
        assert np.shape(Kd) == (3, 3)

        self.attitude_controller_1_Kp = Kp
        self.attitude_controller_1_Kd = Kd

    def set_position_controller_1(self, Kp: np.ndarray, Kd: np.ndarray):
        assert np.shape(Kp) == (3, 3)
        assert np.shape(Kd) == (3, 3)

        self.position_controller_1_Kp = Kp
        self.position_controller_1_Kd = Kd

    ############################################################################################################
    #                                                  Loop                                                    #
    ############################################################################################################

    ##########################################################################
    #                             Sensor Noise                               #
    ##########################################################################

    def simulate_accel_noise(self, accel: np.ndarray):

        biases = self.accel_bias
        stds = self.accel_noise_std

        new_accel = np.array(
            [
                measurement + np.random.normal(bias, std)
                for measurement, std, bias in zip(accel, stds, biases)
            ]
        )

        return new_accel

    def simulate_gyro_noise(self, gyro: np.ndarray):

        biases = self.gyro_bias
        stds = self.gyro_noise_std

        new_gyro = np.array(
            [
                measurement + np.random.normal(bias, std)
                for measurement, std, bias in zip(gyro, stds, biases)
            ]
        )

        return new_gyro

    def simulate_lidar_noise(self, lidar: np.ndarray):

        biases = self.lidar_bias
        stds = self.lidar_noise_std

        new_lidar = np.array(
            [
                measurement + np.random.normal(bias, std)
                for measurement, std, bias in zip(lidar, stds, biases)
            ]
        )

        return new_lidar

    ##########################################################################
    #                               Navigation                               #
    ##########################################################################
    """
        Obtaining current state.
        
        - Adds noise to state variable
        - Runs Kalman filter
    """
    ##########################################################################
    #                                Guidance                                #
    ##########################################################################
    """
        Where the path planning happens.

        - Generate errors based on flight path

    """

    def generate_path_velocity(self, state_desired: np.ndarray):
        """Uses path difference and current velocity heading to generate a desired velocity"""
        pass

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
        dir_for_orientation = self.F_desired

        # Clip to maximum force
        if norm(self.F_desired) > self.max_thrust_N * 1.1:
            self.F_desired = self.F_desired * abs(self.max_thrust_N / norm(self.F_desired))

        # Vertical force fix
        if self.F_desired[2] < 0: # TODO: make better condition for this
        # if angle_between(self.F_desired, [0,0,1]) > max_angle:
            # dir_for_orientation = -(self.F_g/self.F_desired[2]) * self.F_desired
            # dir_for_orientation = np.copy(self.F_desired)
            dir_for_orientation[2] *= -1

            # Mapping to range of minimum thrusts (very proud of this)
            self.F_desired[2] = (self.F_g - self.min_thrust_N) * ( np.arctan(self.F_desired[2] / 2) + np.pi/2 ) + self.min_thrust_N
        # else:
        #     self.F_desired[2] = (self.max_thrust_N - self.F_g) * ( np.arctan(self.F_desired[2] / 2) + np.pi/2 ) + self.F_g

         # Construct orthogonal frame to find desired quaternion
        z_axis_hat = unit(dir_for_orientation)
        x_axis_hat = unit(np.cross(z_axis_hat, np.cross(np.array([1,0,0]), z_axis_hat)) ) # assigns heading based off of desired velocity. TODO: maybe DONT DO THIS

        # x_axis_hat = unit(np.cross(z_axis_hat, np.cross(v_desired_L, z_axis_hat)) ) # assigns heading based off of desired velocity. TODO: maybe DONT DO THIS

        # print(x_axis_hat)
        y_axis_hat = unit(np.cross(z_axis_hat, x_axis_hat))

        R = np.column_stack((x_axis_hat, y_axis_hat, z_axis_hat))
        q_des = unit(quat_from_R(R))
        # q_des = quat_from_R(R)
        # print(R)
        # print(quat_apply(q_des, [0,0,1]))
        # print(norm())
        # breakpoint()


        thrust = norm(self.F_desired)

        # print(R)
        # print(quat_apply(q_des, [0,0,1]))

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

    ##########################################################################
    #                         Sample Controllers                             #
    ##########################################################################

    def z_spin_controller(self, spin: float = 0.02):
        """Returns actuator inputs in a 4-element np.ndarray"""

        F_g = self.F_g

        F_extra = 0  # N

        # Minimum force from the propellers
        min_force = 4 * self.force_bounds_N[0]

        thrusts = (
            self.allocate_thrusts(F_g + F_extra - min_force, np.array([0, 0, spin]))
            + min_force / 4
        )

        return thrusts

    def vertical_sample_controller(self, vertical_angle: float):
        """Returns actuator inputs in a 4-element np.ndarray"""

        # p_z
        kp = 0.15
        kd = 0.5

        if self.t > 50:
            desired = 100
        else:
            desired = 250

        err = desired - self.state[2]

        #                       v_z - v_z_prev
        # additional = - kp * (self.state[3] - prev_state[3])
        additional = kp * err

        # Only after first timestep
        if self.t > self.dt:
            additional += kd * (err - self.prev_p_error) / self.dt

        thrust = self.F_g / np.cos(vertical_angle) + additional

        self.prev_p_error = err

        return thrust  # N


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

        prev_state = self.state

        if self.full_navigation:
            pass
        else:

            
            sim_state = self.get_sim_state() # OR Add option to feed sensor values instead
            self.t += self.dt

            # Calculated state is actual state
            self.p_calc = sim_state[0:3]
            self.v_calc = sim_state[3:6]
            self.q_calc = sim_state[6:10]
            self.w_calc = sim_state[10:13]

            # breakpoint()

        self.state = np.concat([self.p_calc, self.v_calc, self.q_calc, self.w_calc])

        # # Simulated sensor Noise
        # self.a_true = (self.state[3:6] - prev_state[3:6]) / self.dt

        # self.a_noise = self.simulate_accel_noise(self.a_true)
        # self.w_noise = self.simulate_gyro_noise(self.w_true)
        # self.lidar_p_noise = self.simulate_lidar_noise(self.p_true)

        # FIltering
        # a,w -> p,v,q,w


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