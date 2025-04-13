import numpy as np
from .quaternion_helpers import *
from typing import Callable

class Drone:

    def __init__(self, dt: float, state0: np.ndarray = None):

        if dt <= 0:
            raise ValueError("dt must be greater than 0")
        
        if state0 is not None and type(state0) is not np.ndarray:
            raise ValueError("If state0 is input, must be an ndarray")
        # self.state = state0
        self.state = np.zeros(13) # p (3), v (3), q (4), w (3)
        self.dt = dt
        self.t = 0

        # Must initialize everything with functions
        self.mass = None
        self.I_inv = None
        self.dimensions = None

        # Erorrs
        self.prev_angle_error = 0 # rad
        self.prev_error = 0

    def add_sim(self, sim_state_func: Callable[[], np.ndarray], sim_time_func: Callable[[], float]):
        self.get_sim_state = sim_state_func
        self.get_sim_time = sim_time_func

    ############################################################################################################
    #                                        Drone Initialization                                              #
    ############################################################################################################

    def define_prop(self, 
                     arm_distance: float,
                     prop_height: float,
                     max_force: float, 
                     min_force: float,
                     num: int = 4,
                     kd: float = 0.02):
        
        if arm_distance < 0:
            raise ValueError("Arm distance must be positive")

        self.arm_distance = arm_distance
        self.prop_height = prop_height

        self.force_bounds = [min_force * 9.81, max_force * 9.81]
        self.num_prop = num
        self.kd = kd

        # Allocation Matrix
        # Reference: https://www.cantorsparadise.org/how-control-allocation-for-multirotor-systems-works-f87aff1794a2/
        r = arm_distance / np.sqrt(2) # Distance from prop to central axis

        allocation_matrix = np.array([
            [1, 1, 1, 1],
            [r, -r, r, -r],
            [-r, -r, r, r],
            [r, -r, -r, r]
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
        self.dimensions = np.array(dimensions)
        self.I_inv = np.linalg.inv(I)

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

    def get_desired_torque(self, q_B2L: np.ndarray, q_des_L: np.ndarray, gains: list):
        """
        TODO: FINISH THIS IT MAY SUCK

        Frames:
        - L = Launch
        - B = Body
        - D = Desired
        """

        q_B2D = quat_mult( quat_inv(q_des_L), q_B2L)

        angle, axis = axis_rot_from_quat(q_B2D)

        self.prev_angle_error = angle

        factor = gains[0] * angle + gains[1] * (angle - self.prev_angle_error)

        return factor * unit(axis) # Axis might already be normalized
    
    def constant_v_guidance(self, v_des_L: np.ndarray):
        """

        Contains gains for angle correction
        Pseudocode:
        - Config
            - Determine dV, norms of both velocities, and "midpoint" velocity vector/norm
        - Running
            - If speed increasing and less than speed of midpoint norm
                - Increase travel angle
            - If speed 
        - While velocity in direction is before (V_0 + dV) - (if )
        
        Increases travel angle (facing forward)
        """
        q_B2L = self.q

        # Torque correction gains
        kp_t = 0.01 # Nm/rad
        kd_t = 0.001 # Nm•s/rad

        gains = [kp_t, kd_t, 0]

        torque = self.get_desired_torque(q_B2L, )

    ##########################################################################
    #                                Controls                                #
    ##########################################################################   
    """
        
    
        - Create desired force/torque with controllers
        - Allocates desired force/torque to propellers
    """ 

    

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
        F_g = self.mass * 9.81

        F_extra = 0 # N

        # Minimum force from the propellers
        min_force = 4 * self.force_bounds[0]

        thrusts = self.allocate_thrusts(F_g + F_extra - min_force, np.array([0.00001,0,.00001])) + min_force/4
        
        
        return thrusts 

    def allocate_thrusts(self, thrust_z: float, torques: np.ndarray) -> np.ndarray:
        # Reference:
        # https://www.cantorsparadise.org/how-control-allocation-for-multirotor-systems-works-f87aff1794a2/

        outputs = np.concat(([thrust_z], torques))
        inputs = np.matmul(self.A_inv, outputs)

        return inputs      

    def apply_motor_bounds(self, commands: np.ndarray):

        result = np.clip(commands, a_min=self.force_bounds[0], a_max=self.force_bounds[1])
        # print(result)
        return result
    
    # def thrust_to_speed_command(self)

    ##########################################################################
    #                                 Loop                                   #
    ##########################################################################
    """
        - Runs all GNC algorithms
    """ 

    def timestep(self):

        ######### Navigation #########

        prev_state = self.state
        self.state = self.get_sim_state()
        self.t += self.dt

        self.p = self.state[0:3]
        self.v = self.state[3:6]
        self.q = self.state[6:10]
        self.w = self.state[10:13]

        ######### Guidance #########


        ######### Control #########
        # Analyze forces
        # Set motor_forces

        

        # v_z
        # kp = 0.05
        # kd = 0.09
        # desired = 5
        # err = (self.state[5] - desired)

        # self.motor_forces = self.vertical_sample_controller()
        self.motor_forces = self.force_up_sample_controller()




        
        # print(self.motor_forces, end="\t\t\t")
        # print(sum(self.motor_forces), end="\t\t\t")
        # print(self.state[10:13])
              

        # Command motors based on lookup tables

        ######### Propogation? #########
        
        
