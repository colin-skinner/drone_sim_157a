from dronesim import (
    Simulation,
    Drone,
    Logger,
    quat_apply,
    unit,
    plot_state_vector,
    plot_drone_axis,
    plot_1, plot_3, plot_3d, debug_3d
)

import parameters as p
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

"""
TODO:
- Fix allocation matrix and motor definitions
- Better attitude controller
- Better position controller
- Spline trajectory

"""


if __name__ == "__main__":



    ########################################
    #               Objects                #
    ########################################

    drone = Drone(p.dt, p.state0)
    sim = Simulation(p.t_max, p.dt, p.state0)

    # Physical Properties
    drone.define_prop(70 / 1000, 15 / 1000, p.max_prop_force_kgf, p.min_prop_force_kgf, 0)
    drone.define_drone(p.mass, p.I, p.dimensions)

    # Simulation Properties
    # sim.add_imu_misalignment(p.imu_misalignment)
    sim.add_accel_noise(p.accel_bias, p.accel_std)
    sim.add_gyro_noise(p.gyro_bias, p.gyro_std)
    sim.add_lidar_noise(p.lidar_bias, p.lidar_std)

    sim.add_drone(drone)
    drone.add_sim_functions(sim.get_state, sim.get_time)
    drone.make_ekf(p.P0, p.accel_bias, p.gyro_bias, p.lidar_bias)
    drone.add_navigation_data_functions(sim.generate_navigation_data, p.drone_full_navigation)

    ########################################
    #                Path                  #
    ########################################

    drone.add_path(p.p_d_arr)

    ########################################
    #               Gains                  #
    ########################################

    drone.set_attitude_controller_1(
        np.diag(p.attitude_controller_1_kp),
        np.diag(p.attitude_controller_1_kd),
    )

    drone.set_position_controller_1(
        np.diag(p.position_controller_1_kp),
        np.diag(p.position_controller_1_kd),
    )

    ########################################
    #               Logger                 #
    ########################################

    logger = Logger(p.t_max, p.dt)
    logger.add_drone(drone)
    logger.add_sim(sim)

    ########################################
    #             Simulate                 #
    ########################################

    step = 0
    while sim.t < p.t_max:
        sim.sim_drone_timestep()
        logger.log(step)
        step += 1

        if sim.actual_state[2] < 0 or drone.dead:
            # if sim.drone.state[2] < 0:
            print("U CRASHED")
            break

    # Set ending step
    logger.step = step

    if p.filename not in ["", None]:
        logger.save(p.filename)
        logger.save_kalman(p.filename)


    ########################################
    #             Analysis                 #
    ########################################

    plot_state_vector(logger)
    plot_3(logger.t[:step], logger.ekf_state[:step, 0:3], "EKF Position")
    plot_3(logger.t[:step], logger.ekf_state[:step, 3:6], "EKF Velocity")

    plot_drone_axis(logger, [0,0,1], "Drone Normal Vector")
    plot_drone_axis(logger, [1,0,0], "Drone X Vector")

    plot_1(logger.t[:step], logger.drone_commanded_thrust[:step], "Drone Commanded Thrust")

    plot_3(logger.t[:step], logger.drone_commanded_torques[:step,:], "Drone Commanded Torques")

    plot_1(logger.t[0:step], logger.drone_vertical_angle[:step], "Drone Vertical Angle")

    plot_3(logger.t[0:step], logger.actual_forces[:step], "Drone Actual Forces")

    plot_3(logger.t[0:step], logger.actual_torques[:step,:], "Drone Actual Torques")

    # plot_3(logger.t[:step], logger.actual_a_body[:step,:], "a body")
    # plot_3(logger.t[:step], logger.actual_w_body[:step,:], "w body")

    plot_3(logger.t[:step], drone.a_body_array, "a body drone")
    plot_3(logger.t[:step], drone.w_body_array, "w body drone")


    

    plot_3(logger.t[:step], [unit(quat_apply(q_d, [0,0,1])) for q_d in logger.drone_desired_quat[:step,:]], "q_des")


    # plot_vec_3d(ax, curr_p, curr_p + unit(quat_apply(q_d, [0,0,1])), 'black')


    plot_3d(logger)


    if p.DEBUG:

        plt.show(block=False)
        breakpoint()
        debug_3d(logger, figsize=(10,10), start_time_s=p.debug_start_time, interval=p.speed_interval)
        breakpoint()
    else: 
        plt.show()
