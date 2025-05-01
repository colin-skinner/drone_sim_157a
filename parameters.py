import numpy as np
from dronesim import quat_from_axis_rot, ThrustData


########################################
#           Initial State              #
########################################

p0_m = [0, 0, 2]
# p0_m = [0,5,8]
v0_m = [0, 0, 0]
q0 = [1, 0, 0, 0]  # Identity quaternion
q0 = quat_from_axis_rot(80, [0, 1, 0]).tolist()  # 20 deg angle in y
w0_rad_s = [0, 0, 0]
state0 = np.array(p0_m + v0_m + q0 + w0_rad_s)

########################################
#             Mass Stuff               #
########################################

mass = 1  # kg
I = np.array([[0.00030,         0,              0],
              [0,               0.00030,        0],
              [0,               0,              0.00045]])
dimensions = np.array([13, 13, 8])  # input into list as cm


########################################
#            Prop Stuff                #
########################################

thrust_data = ThrustData("Calibration Data/Motor_Kv1860_Orange_Propeller_Data.xlsx", drop_duplicates=True)
# print(thrust_data.lookup_table)

min_prop_force_kgf = min(thrust_data.lookup_table["Thrust (kgf)"])
max_prop_force_kgf = max(thrust_data.lookup_table["Thrust (kgf)"])

# min_prop_force_kgf = 0.095
# max_prop_force_kgf = 0.46

# print(f"{min_prop_force_kgf=}")
# print(f"{max_prop_force_kgf=}")

# ADDD LOOKUP TABLE PROP

########################################
#               Path                   #
########################################

# p_d_arr = { # testing Z
#     0: [0,0,4],
#     2: [0,0,6],
#     4: [0,0,8],
#     6: [0,0,10],
#     8: [0,0,20],
#     10:[0,0,2],
# }

p_d_arr = { # testing Z
    0: ([0,0,2],[0,0,0]),
    2: ([0,0,4],[0,0,0]),
    4: ([0,0,6],[0,0,0]),
    6: ([0,0,8],[0,0,0]),
    8: ([0,0,10],[0,0,0]),
    10: ([0,0,5],[0,0,0])
}

p_d_arr = { # testing Z
    0: ([0,0,2],[0,0,0]),
    2: ([0,1,4],[0,0,1]),
    4: ([0,2,6],[0,0,0.2]),
    6: ([0,3,8],[0,0,0.4]),
    8: ([0,8,10],[0,0,0.6]),
    12: ([0,5,5],[0,0,0])
}



# p_d_arr = { # testing X
#     0: [0,0,2],
#     2: [1,0,2],
#     4: [2,0,2],
#     6: [3,0,2],
#     8: [4,0,2],
# }


# p_d_arr = { # testing Y

#     0: ([0,1,2],[0,0,0]),
#     4: ([0,2,2],[0,0,0]),
#     6: ([0,3,2],[0,0,0]),
#     8: ([0,4,2],[0,0,0])
# }

# p_d_arr = { # weird one
#     0: ([3,3,5], [0, 0, 0]),
#     2: ([0,5,8], [0, 0, 0]),
#     5: ([2,3,6], [0, 0, 0])
# }

# p_d_arr = { # cool alternating one
#     0: ([0,5,8], [0, 0, 0]),
#     3: ([2,3,6], [0, 0, 0]),
#     6: ([8,5,8], [0, 0, 0]),
#     9: ([6,3,6], [0, 0, 0]),
#     12: ([0,5,8], [0, 0, 0]),
#     15: ([5,3,8], [0, 0, 0]),
#     18: ([8,5,8], [0, 0, 0]),
#     22: ([2,3,8], [0, 0, 0]),
#     25: ([2,3,5], [0, 0, 0])
# }

# p_d_arr = {
#     0: (p0_m, [0, 0, 0]),
#     0: ([5,3,8], [0, 0, 0]),
# }


########################################
#             Simulation               #
########################################

t_max = 20
dt = 0.001

imu_misalignment = [1,0,0,0]

accel_bias = [0,0,0]
accel_std = [0,0,0]

gyro_bias = [0,0,0]
gyro_std = [0,0,0]

lidar_bias = [0,0,0]
lidar_std = [0,0,0]

filename = "KALMAN_TEST"

DEBUG = True
# DEBUG = False

debug_start_time = 7.95   # Seconds into sim to start
# debug_start_time = 11.9
speed_interval = 25    # Frames to travel at once for 0.001 FAST
speed_interval = 15    # Frames to travel at once for 0.001 SLOW
# speed_interval = 2    # Frames to travel at once for 0.01

########################################
#         Controller Gains             #
########################################

attitude_controller_1_kp = 3 * [0.6] # GOOD and somewhat related to last row of allocation matrix for kd*2*r
attitude_controller_1_kd = 3 * [0.008] # GOOD


attitude_controller_1_kp = 3 * [1.0] # GOOD and somewhat related to last row of allocation matrix for kd*r
attitude_controller_1_kd = 3 * [0.025] # GOOD


# attitude_controller_1_kp = [0.3, 0.3, 0.6]
# attitude_controller_1_kd = [0.004, 0.004, 0.008]

# attitude_controller_1_kp = [0.3, 0.3, 0.3]
# attitude_controller_1_kd = [0.004, 0.004, 0.004]
# attitude_controller_1_kd = -0.2 * np.sqrt(attitude_controller_1_kp)

# position_controller_1_kp = 3 * [0.1]
# position_controller_1_kd = 3 * [-0.6]

# a = 3
# b = 0.4
a = 10.5
b = 5.5
position_controller_1_kp = [12.5, 12.5, 9.5] # good Z
position_controller_1_kd = [6.5, 6.5, 5.2] # good Z

position_controller_1_kp = 3 * [a]
position_controller_1_kd = 3 * [b]




