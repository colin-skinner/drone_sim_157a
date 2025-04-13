# from simulation import Simulation
# from drone import Drone
# from logger import Logger
# from quaternion_helpers import quat_apply, quat_inv

from dronesim import Simulation, Drone, Logger, quat_apply, quat_inv
import numpy as np
# import pandas as pd

import matplotlib.pyplot as plt
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

    p0 = [0,0,10]
    v0 = [0,0,0]
    q0 = [1,0,0,0] # Identity quaternion
    w0 = [0,0,0]
    state0 = np.array(p0 + v0 + q0 + w0)

    mass = .5 # kg
    I = np.array([
        [0.00030, 0, 0],
        [0, 0.00030, 0],
        [0, 0, 0.00045]
    ])
    dimensions = np.array([13, 13, 8]) # input into list as cm

    min_prop_force_kgf = 0.095
    max_prop_force_kgf = 0.46

    # ADDD LOOKUP TABLE PROP

    t_max = 60
    dt = 0.01
    ########################################
    #               Objects                #
    ########################################
    

    drone = Drone(dt, state0)
    sim = Simulation(t_max, dt, state0)



    
    drone.define_prop(70/1000, 15/1000, max_prop_force_kgf, min_prop_force_kgf, 0)
    drone.define_drone(mass, I, dimensions / 100)


    sim.add_drone(drone)
    drone.add_sim(sim.get_state, sim.get_time)

    ########################################
    #               Logger                #
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

        if sim.actual_state[2] < 0:
            print("FUCKASS U CRASHED")
            break
        logger.log(step)

        step += 1

    ########################################
    #             Analysis                 #
    ########################################


    # for row in logger.actual_states[1:step,:].tolist():
    #     print(row)

    plt.figure()
    plt.plot(logger.t[0:step], logger.actual_states[0:step, 0:3])
    plt.title("Position")
    plt.legend(["X", "Y", "Z"])

    plt.figure()
    plt.plot(logger.t[0:step], logger.actual_states[0:step, 3:6])
    plt.title("Velocity")
    plt.legend(["X", "Y", "Z"])

    plt.figure()
    axis = [quat_apply(q, [0,1,0]) for q in logger.actual_states[0:step, 6:10].tolist()]
    plt.plot(logger.t[0:step], axis)
    plt.title("Axis")
    plt.legend(["X", "Y", "Z"])

    plt.show()
    
    print("WHAT")

