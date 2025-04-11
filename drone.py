import numpy as np
from quaternion_helpers import *
from typing import Callable

class Drone:

    def __init__(self, dt: float, state0: np.ndarray):

        # self.state = state0
        self.state = np.zeros(13) # p (3), v (3), q (4), w (3)
        self.dt = dt
        self.t = 0

        # # Force and Torque array in global frame
        # self.force_array: list[np.ndarray] = []
        # self.torque_array: list[np.ndarray] = []

        # Must initialize everything with functions
        self.mass = None
        self.I_inv = None
        self.dimensions = None
        self.prev_error = 0

    def add_sim(self, sim_state_func: Callable[[], np.ndarray], sim_time_func: Callable[[], float]):
        self.get_sim_state = sim_state_func
        self.get_sim_time = sim_time_func

    ########################################
    #      Drone Initialization            #
    ########################################

    def define_props(self, 
                     arm_distance: float,
                     prop_height: float,
                     max_force: float, 
                     min_force: float,
                     num: int = 4):
        
        if arm_distance < 0:
            raise ValueError("Arm distance must be positive")

        self.arm_distance = arm_distance
        self.prop_height = prop_height

        self.force_bounds = [min_force * 9.81, max_force * 9.81]
        self.num_prop = num

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

    ########################################
    #              Allocation              #
    ########################################

    def allocate_thrusts(self, thrust_z: float, torques: np.ndarray) -> np.ndarray:
        # Reference:
        # https://www.cantorsparadise.org/how-control-allocation-for-multirotor-systems-works-f87aff1794a2/

        outputs = np.concat(([thrust_z], torques))
        inputs = np.matmul(self.A_inv, outputs)

        return inputs
    
    # def actual_thrust_torques(self):
    #     """ MIGHT NOT NEED THIS"""
    #     actual_output_forces = self.motor_forces

    #     actual_forces: np.ndarray = np.matmul(self.A, actual_output_forces)

    #     T_actual = float(actual_forces[0])
    #     self.torque_actual = float(actual_forces[1:4])




    ########################################
    #               Controls               #
    ########################################    

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

        thrusts = self.allocate_thrusts(F_g + F_extra - min_force, np.array([1,0,0])) + min_force/4
        
        
        return thrusts 
        # return self.apply_motor_bounds(thrusts)

        

    

    ########################################
    #             Add Forces               #
    ########################################  

    def apply_motor_bounds(self, commands: np.ndarray):

        # assert len(commands) == self.num_prop

        # print(commands, end="\t")

        result = np.clip(commands, a_min=self.force_bounds[0], a_max=self.force_bounds[1])
        # print(result)
        return result

    ########################################
    #               Propogation            #
    ########################################

    def timestep(self):

        ######### Navigation #########

        # Obtain state information. In actual flight or better sims, replaced by sensor readings and filtering
        """
        - Get sensor readings
        - Run EKF to get better p, v, q, w
        """
        prev_state = self.state
        
        self.state = self.get_sim_state() # Add some noise maybe
        self.t += self.dt

        ######### Guidance #########

        # Consult flight path and generate errors
        # errors = desired - actual


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
        
        
        # self.simulated_torques = self.actual_torques()
        
    



    # def allocate_thrusts(self, desired_force: np.ndarray, desired_torque: np.ndarray):
    #     """Control alg that adds thrust to force and moment arrays.
    #     - In the sim, it accounts for spin up and spin down time"""
    #     # r_thruster = 0.84 # m

    #     # Add to force and moment array (very simple in a sim)
    #     self.force_array.append(desired_force)
    #     self.torque_array.append(desired_torque)



    # def calc_forces(self):
    #     """Sums external forces and adds its own inputs"""
    #     self.total_force = sum(self.forces) # I think this works

    # def calc_torques(self):
    #     self.total_torque = np.array(self.torques)

    # def onboard_flight_deriv(self, state: np.ndarray):

    #     v = state[3:6]
    #     q_B2L = state[6:10]
    #     w = state[10:13]

    #     dpdt = v
    #     dvdt = self.total_force / self.mass
    #     dqdt = quat_mult(q_B2L, [0, *w])
    #     dwdt = np.matmul(self.I_inv, self.torque)

    #     return np.array([*dpdt, *dvdt, *dqdt, *dwdt])