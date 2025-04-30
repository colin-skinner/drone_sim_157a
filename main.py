from dronesim import *
# (
#     Simulation,
#     Drone,
#     Logger,
#     quat_apply,
#     quat_inv,
#     quat_from_axis_rot,
#     angle_between,
#     plot_state_vector,
#     plot_drone_axis,
#     plot_1, plot_2, plot_3,
#     RAD2DEG, DEG2RAD
# )

from parameters import *
import numpy as np
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
    #               Objects                #
    ########################################

    drone = Drone(dt, state0)
    sim = Simulation(t_max, dt, state0)

    # Physical Properties
    drone.define_prop(70 / 1000, 15 / 1000, max_prop_force_kgf, min_prop_force_kgf, 0)
    drone.define_drone(mass, I, dimensions / 100)

    # Simulation Properties
    drone.add_imu_misalignment(imu_misalignment)
    drone.add_accel_noise(accel_bias, accel_std)
    drone.add_gyro_noise(gyro_bias, gyro_std)
    drone.add_lidar_noise(lidar_bias, lidar_std)

    sim.add_drone(drone)
    drone.add_sim_functions(sim.get_state, sim.get_time)

    ########################################
    #                Path                  #
    ########################################

    drone.add_path(p_d_arr)

    ########################################
    #               Gains                  #
    ########################################

    drone.set_attitude_controller_1(
        np.diag(3 * [attitude_controller_1_kp]),
        np.diag(3 * [attitude_controller_1_kd]),
    )

    drone.set_position_controller_1(
        np.diag(position_controller_1_kp),
        np.diag(position_controller_1_kd),
    )

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

    # Set ending step
    logger.step = step

    print(step)
    print(logger.step)


        

    ########################################
    #             Analysis                 #
    ########################################

    plot_state_vector(logger)

    plot_drone_axis(logger, [0,0,1], "Drone Normal Vector")
    plot_drone_axis(logger, [1,0,0], "Drone X Vector")

    plot_1(logger.t[:step], logger.drone_commanded_thrust[:step], "Drone Commanded Thrust")

    plot_3(logger.t[:step], logger.drone_commanded_torques[:step,:], "Drone Commanded Torques")

    plot_1(logger.t[0:step], logger.drone_vertical_angle[:step], "Drone Vertical Angle")

    plot_3(logger.t[0:step], logger.actual_forces[:step], "Drone Actual Forces")

    plot_3(logger.t[0:step], logger.actual_torques[:step,:], "Drone Actual Torques")

    # plt.figure()
    # plt.plot(logger.t[0:step], logger.actual_forces[0:step])
    # plt.title("Actual Forces")
    # plt.legend(["X", "Y", "Z"])
    # plt.legend()

    # plt.figure()
    # plt.plot(logger.t[0:step], logger.actual_torques[0:step])
    # plt.title("Actual Torques")
    # plt.legend()
    # plt.legend(["X", "Y", "Z"])


    plt.show()

    # logger.save("OH")

    # print("WHAT")
