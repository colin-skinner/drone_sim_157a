import numpy as np
from .quaternion_helpers import *
from typing import Callable

class Drone:

    def __init__(self, dt: float, state0: np.ndarray = None):

        if dt <= 0:
            raise ValueError("dt must be greater than 0")
        
        if state0 is not None and type(state0) is not np.ndarray:
            raise ValueError("If state0 is input, must be an ndarray")
        self.state = state0
        self.fsm_state = "idle"
        self.dt = dt
        self.t = 0
        

        # Must initialize everything with functions
        self.mass = None
        self.F_g = None
        self.I_inv = None
        self.dimensions = None
        self.dead = False

        # Erorrs
        self.prev_angle_error = 0 # rad
        self.prev_error = 0

    def add_sim_functions(self, sim_state_func: Callable[[], np.ndarray], sim_time_func: Callable[[], float]):
        self.get_sim_state = sim_state_func
        self.get_sim_time = sim_time_func

    ############################################################################################################
    #                                       Sensor Initialization                                              #
    ############################################################################################################

    def add_accel_noise(self,
                       bias: np.ndarray = np.zeros(3), 
                       noise_mean: np.ndarray = np.zeros(3), 
                       noise_std: np.ndarray = np.zeros(3)):

        if len(bias) != 3:
            raise ValueError("Length of accelerometer bias array must be 3")
        
        if len(noise_mean) != 3:
            raise ValueError("Length of accelerometer noise mean array must be 3")
        
        if len(noise_std) != 3:
            raise ValueError("Length of accelerometer standard deviation array must be 3")
        
        self.accel_bias = bias
        self.accel_noise_mean = noise_mean
        self.accel_noise_std = noise_std

    def add_gyro_noise(self,
                       bias: np.ndarray = np.zeros(3), 
                       noise_mean: np.ndarray = np.zeros(3), 
                       noise_std: np.ndarray = np.zeros(3)):

        if len(bias) != 3:
            raise ValueError("Length of gyroscope bias array must be 3")
        
        if len(noise_mean) != 3:
            raise ValueError("Length of gyroscope noise mean array must be 3")
        
        if len(noise_std) != 3:
            raise ValueError("Length of gyroscope standard deviation array must be 3")
        
        self.gyro_bias = bias
        self.gyro_noise_mean = noise_mean
        self.gyro_noise_std = noise_std

    def add_lidar_noise(self,
                       bias: np.ndarray = np.zeros(3), 
                       noise_mean: np.ndarray = np.zeros(3), 
                       noise_std: np.ndarray = np.zeros(3)):
        if len(bias) != 3:
            raise ValueError("Length of lidar bias array must be 3")
        
        if len(noise_mean) != 3:
            raise ValueError("Length of lidar noise mean array must be 3")
        
        if len(noise_std) != 3:
            raise ValueError("Length of lidar standard deviation array must be 3")
        
        self.lidar_bias = bias
        self.lidar_noise_mean = noise_mean
        self.lidar_noise_std = noise_std

    ############################################################################################################
    #                                        Drone Initialization                                              #
    ############################################################################################################

    def define_prop(self, 
                     arm_distance: float,
                     prop_height: float,
                     max_force_kgf: float, 
                     min_force_kgf: float,
                     num: int = 4,
                     kd: float = 0.02):
        
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
        r = arm_distance / np.sqrt(2) # Distance from prop to central axis

        allocation_matrix = np.array([
            [1, 1, 1, 1],
            [-r, -r, r, r],
            [-r, r, -r, r],
            [kd, -kd, -kd, kd]
        ])

        # allocation_matrix = np.array([
        #     [1, 1, 1, 1],
        #     [-r, r, -r, r],
        #     [-r, -r, r, r],
        #     [kd * r, -kd * r, -kd * r, kd * r]
        # ])


        # SEEMED TO WORK FOR THE CONTROLLER
        allocation_matrix = np.array([
            [1, 1, 1, 1],
            [r, -r, r, -r],
            [-r, -r, r, r],
            # [kd * r, -kd * r, -kd * r, kd * r]
            [kd, -kd, -kd, kd]
        ])

        self.A = allocation_matrix
        self.A_inv = np.linalg.inv(allocation_matrix)

    def define_drone(self, mass: float, I: np.ndarray[float], dimensions: list[float]):
        
        if mass <= 0:
            raise ValueError("Mass must be greater than 0 kg")
        
        if np.shape(I) != (3,3):
            raise ValueError("I matrix must be 3x3")
        
        if len(dimensions) != (3):
            raise ValueError("dimensions must be a 3-element list with X,Y,Z lengths")
        
        self.mass = mass
        self.F_g = (9.81 * self.mass)
        self.dimensions = np.array(dimensions)
        self.I = I
        self.I_inv = np.linalg.inv(I)

    ############################################################################################################
    #                                         GNC Initialization                                               #
    ############################################################################################################

    def set_attitude_controller_1(self, Kp: np.ndarray, Kd: np.ndarray):
        assert np.shape(Kp) == (3,3)
        assert np.shape(Kd) == (3,3)

        self.attitude_controller_1_Kp = Kp
        self.attitude_controller_1_Kd = Kd

    ############################################################################################################
    #                                                  Loop                                                    #
    ############################################################################################################

    ##########################################################################
    #                             Sensor Noise                               #
    ##########################################################################

    def simulate_accel_noise(self, accel: np.ndarray):
        
        biases = self.accel_bias
        means = self.accel_noise_mean
        stds = self.accel_noise_std

        new_accel = np.array([measurement + np.random.normal(mean, std) + bias
                              for measurement, mean, std, bias 
                              in zip(accel, means, stds, biases)])

        return new_accel
    
    def simulate_gyro_noise(self, gyro: np.ndarray):
        
        biases = self.gyro_bias
        means = self.gyro_noise_mean
        stds = self.gyro_noise_std

        new_gyro = np.array([measurement + np.random.normal(mean, std) + bias
                              for measurement, mean, std, bias 
                              in zip(gyro, means, stds, biases)])

        return new_gyro
    
    def simulate_lidar_noise(self, lidar: np.ndarray):
        
        biases = self.lidar_bias
        means = self.lidar_noise_mean
        stds = self.lidar_noise_std

        new_lidar = np.array([measurement + np.random.normal(mean, std) + bias
                              for measurement, mean, std, bias 
                              in zip(lidar, means, stds, biases)])

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

    # def get_desired_torque(self, q_B2L: np.ndarray, q_des_L: np.ndarray, gains: list):
    #     """
    #     TODO: FINISH THIS IT MAY SUCK

    #     Frames:
    #     - L = Launch
    #     - B = Body
    #     - D = Desired
    #     """

    #     q_B2D = quat_mult( quat_inv(q_des_L), q_B2L)

    #     angle, axis = axis_rot_from_quat(q_B2D)

    #     self.prev_angle_error = angle

    #     factor = gains[0] * angle + gains[1] * (angle - self.prev_angle_error)

    #     return factor * unit(axis) # Axis might already be normalized
       
    def generate_path_velocity(self,  state_desired: np.ndarray):
        """Uses path difference and current velocity heading to generate a desired velocity"""
        pass

    ##########################################################################
    #                                Controls                                #
    ##########################################################################   
    """
        - Create desired force/torque with controllers
        - Allocates desired force/torque to propellers
    """ 

    def generate_v_command(self, v_desired_L: np.ndarray):
        """Adjusts accelerations to nullify velocity error"""
        gains = [0.01, 0.01, 0] # kp, kd, ki

        accel_L = np.zeros(3)

        return accel_L
    
    # def generate_thrust_torques(self, a_desired_L: np.ndarray):
    #     """Generates body thrust and torques to nullify acceleration error"""
    #     thrust = 0.0
    #     torques = np.zeros(3)

    #     q_B2L = self.q_calc
    #     q_L2B = quat_inv(q_B2L)

    #     # Total acceleration desired
    #     a_tot_L = a_desired_L + np.array([0, 0, 9.81])

    #     # Get normal axis to drone
    #     normal_vec = quat_apply(q_B2L, [0,0,1])
    #     # print(normal_vec)
        
    #     # n x a = desired rotation axis
    #     desired_axis_L = np.cross(normal_vec, a_tot_L)
    #     print(desired_axis_L)
    #     desired_axis_B = quat_apply(q_L2B, desired_axis_L) * 0.0001

    #     # Drives error between desired axis and w axis to 0, finding torque axis
    #     w_L = self.w_calc

    #     if norm(unit(w_L)) < 0.001:
    #         torque_axis_L = desired_axis_L
    #     else:
    #         torque_axis_L = unit(np.cross(w_L, desired_axis_L)) * norm(desired_axis_L)

    #     # Amount of torque determined by PID and 
    #     angle_error = angle_between(normal_vec, desired_axis_L)

    #     print(angle_error * 180 / np.pi)

    #     kp = 100
    #     kd = 100

    #     factor = kp * angle_error + kd * (angle_error - self.prev_angle_error) / self.dt

    #     torques_B = desired_axis_L + torque_axis_L * factor

    #     # Ensures thrust vertical component equal to gravity
    #     thrust_B = self.F_g / normal_vec[2]

    #     thrust_B = np.clip(thrust_B, a_min=self.min_thrust_N, a_max=self.max_thrust_N)

    #     # print(thrust_B / self.mass)
    #     # breakpoint()

    #     self.prev_angle_error = angle_error

    #     return thrust_B, torques_B
    
    def attitude_controller_1(self, q_desired_L: np.ndarray, w_desired_L: np.ndarray) -> np.ndarray:
        assert np.shape(q_desired_L) == (4,)
        assert np.shape(w_desired_L) == (3,)

        # breakpoint()

        q_error_L = quat_mult( quat_inv(q_desired_L), self.q_calc)

        axis_rot_from_quat(q_error_L)
        # print(quat_apply(q_error_L, [0,0,1]))
        w_error_L = w_desired_L - self.w_calc  
        # w_error_L = self.w_calc - w_desired_L

        kp = self.attitude_controller_1_Kp
        kd = self.attitude_controller_1_Kd

        # print(q_error_L[0] * np.matmul(kp, q_error_L[1:4].transpose()))
        # breakpoint()

        torque_L = - q_error_L[0] * np.matmul(kp, q_error_L[1:4].transpose()) - np.matmul(kd, w_error_L.transpose())

        # if abs(torque_L[2]) > 0.002:
        #     breakpoint() 
        # print(torque_L)
        # breakpoint()
        return torque_L 
    
    
    def allocate_thrusts(self, thrust_z_B: float, torques_B: np.ndarray) -> np.ndarray:
        # Reference:
        # https://www.cantorsparadise.org/how-control-allocation-for-multirotor-systems-works-f87aff1794a2/

        outputs = np.concat(([thrust_z_B], torques_B))
        inputs = np.matmul(self.A_inv, outputs)

        # print(inputs)
        # breakpoint()

        return inputs      

    def apply_motor_bounds(self, commands: np.ndarray):

        result = np.clip(commands, a_min=self.force_bounds_N[0], a_max=self.force_bounds_N[1])
        return result
    
    ##########################################################################
    #                         Sample Controllers                             #
    ##########################################################################      

    def z_spin_controller(self, spin: float = 0.02):
        """Returns actuator inputs in a 4-element np.ndarray"""

        F_g = self.F_g

        F_extra = 0 # N

        # Minimum force from the propellers
        min_force = 4 * self.force_bounds_N[0]

        thrusts = self.allocate_thrusts(F_g + F_extra - min_force, np.array([0,0,spin])) + min_force/4
        
        # print(thrusts)
        # print(sum(thrusts))
        # breakpoint()
        
        return thrusts 
    
    def vertical_sample_controller(self):
        """Returns actuator inputs in a 4-element np.ndarray"""

        prop_force = self.mass/4
        
        # p_z
        kp = .035
        kd = .03

        if self.t > 10 and self.t % 20 < 10:
            desired = 15
        else:
            desired = 10

        err = (desired - self.state[2])

        #                       v_x - v_x_prev
        # additional = - kp * (self.state[3] - prev_state[3])
        # err = (self.state[2] - desired)
        additional = kp * err

        # Only after first timestep
        if self.t > self.dt:
            additional += kd * (err - self.prev_error) / self.dt

        self.prev_error = err
        
        motor_cmds = np.array([prop_force + additional, prop_force + additional, prop_force + additional, prop_force + additional]) 

        return self.apply_motor_bounds(motor_cmds * 9.81) # N  

    def force_up_sample_controller(self):
        """Returns actuator inputs in a 4-element np.ndarray"""

        F_g = self.F_g

        F_extra = 20 # N

        # Minimum force from the propellers
        min_force = 4 * self.force_bounds_N[0]

        thrusts = self.allocate_thrusts(F_g + F_extra - min_force, np.array([0,0,.00001])) + min_force/4
        
        
        return thrusts 

    def attitude_tracking_controller(self, q_desired_L: np.ndarray, w_desired_L):

        assert len(q_desired_L) == 4
        assert len(w_desired_L) == 3

        thrust = 0.0
        torques = np.zeros(3)

        q_B2L = self.q_calc
        q_L2B = quat_inv(q_B2L)

    ############################################################################################################
    #                                                Running                                                   #
    ############################################################################################################

    def state_machine(self):

        match self.fsm_state:

            case "idle":
                # Hovers at a specific position with no actions
                
                pass

            case "load_trajectory":
                # Hovers at a specific position with path planning
                # Sends trajectory to operator
                pass

            case "await_confirmation":
                # Idle, but awaiting confirmation from operator
                pass

            case "fly":
                # 💃
                pass

            case _:
                pass
    
    def timestep(self):


        ####### State Machine ########
        """Governs which controller/trajectory to follow"""
        
        ######### Navigation #########

        prev_state = self.state
        self.state = self.get_sim_state()
        self.t += self.dt

        # Actual state
        self.p_true = self.state[0:3]
        self.v_true = self.state[3:6]
        self.q_true = self.state[6:10]
        self.w_true = self.state[10:13]

        # Simulated sensor Noise
        self.a_true = (self.state[3:6] - prev_state[3:6]) / self.dt

        # self.a_noise = self.simulate_accel_noise(self.a_true)
        # self.w_noise = self.simulate_gyro_noise(self.w_true)
        # self.lidar_p_noise = self.simulate_lidar_noise(self.p_true)

        # FIltering
        # a,w -> p,v,q,w
        
        # Calculated state
        self.p_calc = self.p_true   
        self.v_calc = self.v_true
        self.q_calc = self.q_true
        self.w_calc = self.w_true


        ######### Guidance #########


        ######### Control #########
        

        # Analyze forces
        # Set motor_forces

        self.motor_forces = np.zeros(4)
        # self.motor_forces = self.force_up_sample_controller()
        # self.motor_forces += self.z_spin_controller(0.00002)
        # self.motor_forces += self.vertical_sample_controller()
        # thrust, torques = self.generate_thrust_torques(np.array([0.01,0,0]))
        # self.motor_forces += self.allocate_thrusts(thrust, torques)

        q_d = np.array([1,0,0,0])
        w_d = np.zeros(3)
        w_d = np.array([0,0,.0001])

        torques = self.attitude_controller_1(q_d, w_d)
        error_angle = angle_between(self.q_calc, q_d)
        vertical_axis = quat_apply(self.q_calc, [0,0,1])
        vertical_angle = angle_between(vertical_axis, [0,0,1])
        vertical_angle = angle_between(self.q_calc, [1,0,0,0])

        if error_angle < 0.000001:
            torques = np.array(np.zeros(3))
            
        # print(error_angle * 180 / np.pi) 
        # print(torques)


        thrust = self.F_g / np.cos(vertical_angle)
        # print(thrust)
        # stepping = False
        # if (abs(torques[2]) > 0.0000001
        #     or stepping):
        #     # stepping = True
        #     breakpoint()


        thrust = np.clip(thrust, a_min=self.min_thrust_N, a_max=self.max_thrust_N)

        if vertical_angle * 180 / np.pi >= 89:
            self.dead = True


        self.motor_forces += self.apply_motor_bounds(self.allocate_thrusts(thrust, torques))


        self.torques = torques
        self.thrust = thrust
        self.vertical_angle = vertical_angle

        # self.torques = np.zeros(3)
        # self.thrust = 0

        # self.torques = np.zeros(3)
        # self.thrust = 0
        
        # print(self.motor_forces, end="\t\t\t")
        # print(sum(self.motor_forces), end="\t\t\t")
        # print(self.state[10:13])
              

        # Command motors based on lookup tables

        ######### Propogation? #########
        
        
