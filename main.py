# from simulation import Simulation
# from drone import Drone
# from logger import Logger
# from quaternion_helpers import quat_apply, quat_inv

from dronesim import Simulation, Drone, Logger, quat_apply, quat_inv, quat_from_axis_rot, angle_between, RAD2DEG, DEG2RAD
import numpy as np

# import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
"""
- Each timestep
    - Simulation class adds forces to the drone's external force/location list in drone frame
    - Force function sums over the list and converts to inertial frame
    - torque function sums over the list and converts to inertial frame

Flow of program
- Drone Calculation and logging (can be ported to FSW)
    - Navigation: Drone sends current state to Simulation 
    - Guidance: Drone calculates its errors and sends to Simulation
    - Control: Drone acts on control algorithms to generate desired force and torque and sends to Simulation
- Propogation
    - Simulation adds forces and torques to with `add_body_force()`
    - Drone adds input force/torque to its arrays
    - Propogate with RK4

Future
- Flight Path
- Simulating sensor readings for drone to act upon
- Noise in actuator outputs
"""


    


if __name__ == "__main__":

    ########################################
    #               Intial                 #
    ########################################

    p0_m = [0,0,200] 
    v0_m = [0,0,0]
    q0 = [1,0,0,0] # Identity quaternion
    q0 = quat_from_axis_rot(10, [1,1,0]).tolist() # 20 deg angle in y
    w0_rad_s = [0,0,0]
    state0 = np.array(p0_m + v0_m + q0 + w0_rad_s)

    mass = .5 # kg
    I = np.array([
        [0.00030, 0, 0],
         [0, 0.00030, 0],
         [0, 0, -0.00045]
    ])
    dimensions = np.array([13, 13, 8]) # input into list as cm

    min_prop_force_kgf = 0.095
    max_prop_force_kgf = 0.46

    # ADDD LOOKUP TABLE PROP

    t_max = 100
    dt = 0.01

    ########################################
    #               Objects                #
    ########################################
    

    drone = Drone(dt, state0)
    sim = Simulation(t_max, dt, state0)



    # Physical Properties
    drone.define_prop(70/1000, 15/1000, max_prop_force_kgf, min_prop_force_kgf, 0)
    drone.define_drone(mass, I, dimensions / 100)

    # Simulation Properties
    # drone.add_accel_noise()
    # drone.add_gyro_noise()
    # drone.add_lidar_noise()


    sim.add_drone(drone)
    drone.add_sim_functions(sim.get_state, sim.get_time)

    ########################################
    #               Gains                  #
    ########################################


    attitude_controller_1_kp = .03
    attitude_controller_1_kd = 0.004

    # attitude_controller_1_kp = .00003
    # attitude_controller_1_kd = 0.000001



    drone.set_attitude_controller_1(np.diag(3 * [attitude_controller_1_kp]), np.diag(3 * [-attitude_controller_1_kd]))

    ########################################
    #               Logger                 #
    ########################################

    logger = Logger(t_max, dt)
    logger.add_drone(drone)
    logger.add_sim(sim)

    ########################################
    #             Simulate                 #
    ########################################

    step = 0
    while sim.t < t_max:
        sim.sim_drone_timestep()
        logger.log(step)
        step += 1

        if sim.actual_state[2] < 0 or drone.dead: 
        # if sim.drone.state[2] < 0: 
            print("FUCKASS U CRASHED")
            break
        

    ########################################
    #             Analysis                 #
    ########################################


    # for row in logger.actual_states[1:step,:].tolist():
    #     print(row)
    # axs: np.ndarray[Axes]
    # fig, axs = plt.subplots(2, 3)
    # fig.set_figheight(10)
    # fig.set_figwidth(20)

    # axis: Axes = axs[0][0]
    # axis.plot(logger.t[0:step], logger.actual_states[0:step, 0:3])
    # axis.set_title("Position")
    # axis.legend(["X", "Y", "Z"])

    # axis: Axes = axs[0][1]
    # axis.plot(logger.t[0:step], logger.actual_states[0:step, 3:6])
    # axis.set_title("Velocity")
    # axis.legend(["X", "Y", "Z"])

    # dir = [0,1,0]
    # dir = [0,0,1]
    # axis: Axes = axs[1][0]
    # axis_vec = [quat_apply(q_B2L, dir) for q_B2L in logger.actual_states[0:step, 6:10].tolist()]
    # axis.plot(logger.t[0:step], axis_vec)
    # axis.set_title("Axis")
    # axis.legend(["X", "Y", "Z"])

    # axis: Axes = axs[1][1]
    # axis.plot(logger.t[0:step], logger.drone_commanded_torques[0:step, :])
    # axis.set_title("Torques")
    # axis.legend(["X", "Y", "Z"])

    # axis: Axes = axs[0][2]
    # axis.plot(logger.t[0:step], logger.drone_commanded_thrust[0:step])
    # axis.set_title("Thrust")
    # axis.legend()

    # # Vertical angles

    # axis: Axes = axs[1][2]

    # vertical_angle = [angle_between([0,0,1], v) * RAD2DEG for v in axis_vec]
    # axis.plot(logger.t[0:step], vertical_angle)
    # axis.set_title("Vertical angle")
    # axis.legend()


    plt.figure()
    plt.plot(logger.t[0:step], logger.actual_states[0:step, 0:3])
    plt.title("Position")
    plt.legend(["X", "Y", "Z"])

    plt.figure()
    plt.plot(logger.t[0:step], logger.actual_states[0:step, 3:6])
    plt.title("Velocity")
    plt.legend(["X", "Y", "Z"])

    dir = [0,1,0]
    dir = [0,0,1]
    plt.figure()
    axis_vec = [quat_apply(q_B2L, dir) for q_B2L in logger.actual_states[0:step, 6:10].tolist()]
    plt.plot(logger.t[0:step], axis_vec)
    plt.title("Axis")
    plt.legend(["X", "Y", "Z"])

    plt.figure()
    plt.plot(logger.t[0:step], logger.drone_commanded_torques[0:step])
    plt.title("Torques")
    plt.legend(["X", "Y", "Z"])

    plt.figure()
    plt.plot(logger.t[0:step], logger.drone_commanded_thrust[0:step])
    plt.title("Thrust")
    plt.legend()

    plt.figure()
    plt.plot(logger.t[0:step], logger.actual_forces[0:step])
    plt.title("Actual Forces")
    plt.legend(["X", "Y", "Z"])
    plt.legend()

    plt.figure()
    plt.plot(logger.t[0:step], logger.actual_torques[0:step])
    plt.title("Actual Torques")
    plt.legend()
    plt.legend(["X", "Y", "Z"])



    # Vertical angles

    plt.figure()

    plt.plot(logger.t[0:step], logger.drone_vertical_angle[0:step] * RAD2DEG)
    plt.title("Vertical angle")
    plt.legend()

    # breakpoint()


    plt.show()

    logger.save("OH")
    
    # print("WHAT")

