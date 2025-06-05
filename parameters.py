"""Parameters"""
import numpy as np
from dronesim import ThrustData, TrajectoryData, CM2M, quat_from_axis_rot


########################################
#           Initial State              #
########################################

trajectory = TrajectoryData("Inputs/Trajectory Data Draft.xlsx", "New1, 45 CCW y")
p0_m = trajectory.data[0][0]

# p0_m = [0.0, 0.0, 2.0]
# p0_m = [1.5, -1.5, 5.5]
# p0_m = [0, -1.5, 5.5]
# p0_m = [0,5,8]
v0_m = [0.0, 0.0, 0.0]
q0 = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
# q0 = quat_from_axis_rot(70, [0, 1, 0]).tolist()  # 20 deg angle in y
w0_rad_s = [0.0, 0.0, 0.0]
state0 = np.array(p0_m + v0_m + q0 + w0_rad_s)

########################################
#             Mass Stuff               #
########################################

mass = 0.8  # kg
I = np.array([[0.01,         0,              0],  # noqa: E741
              [0,               0.01,        0],
              [0,               0,              0.02]])
dimensions = np.array([13, 13, 8]) * CM2M # input into list as cm



########################################
#            Prop Stuff                #
########################################

thrust_data = ThrustData("Inputs/Motor_Kv1860_Orange_Propeller_Data.xlsx")
# print(thrust_data.lookup_table)

min_prop_force_kgf = float(min(thrust_data.lookup_table["Thrust (kgf)"]))
max_prop_force_kgf = float(max(thrust_data.lookup_table["Thrust (kgf)"]))

# min_prop_force_kgf = 0.095
# max_prop_force_kgf = 0.46

# print(f"{min_prop_force_kgf=}")
# print(f"{max_prop_force_kgf=}")

# ADDD LOOKUP TABLE PROP

########################################
#               Kalman                 #
########################################

P0 = np.zeros((10, 10))
P0[0:3, 0:3] = np.eye(3) * 0.05**2                  # p in m
P0[3:6, 3:6] = np.eye(3) * 0.05**2                  # v in m/s
P0[6:10, 6:10] = np.eye(4) * 1e-5                   # q


########################################
#            Sample Path               #
########################################


p_d_arr = trajectory.data

########################################
#            Sample Path               #
########################################

# p_d_arr = { # testing Z
#     0: [0,0,4],
#     2: [0,0,6],
#     4: [0,0,8],
#     6: [0,0,10],
#     8: [0,0,20],
#     10:[0,0,2],
# }

# p_d_arr = { # testing Z
#     0: ([0,0,2],[0,0,0]),
#     2: ([0,0,4],[0,0,0]),
#     4: ([0,0,6],[0,0,0]),
#     6: ([0,0,8],[0,0,0]),
#     8: ([0,0,10],[0,0,0]),
#     10: ([0,0,5],[0,0,0])
# }



# p_d_arr = { # wacky
#     0: ([0,0,2],[0,0,0]),
#     2: ([0,1,4],[0,0,1]),
#     4: ([0,2,6],[0,0,0.2]),
#     6: ([0,3,8],[0,0,0.4]),
#     8: ([0,8,10],[0,0,0.6]),
#     12: ([0,5,5],[0,0,0])
# }



# p_d_arr = { # testing X
#     0: ([0,0,2],[0,0,0]),
#     2: ([1,0,2],[0,0,0]),
#     4: ([2,0,2],[0,0,0]),
#     6: ([3,0,2],[0,0,0]),
#     8: ([4,0,2],[0,0,0])
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

# p_d_arr = { # cool alternating one
#     0: ([0,5,8], [0, 0, 0]),
#     3: ([2,3,6], [0, 0, 0]),
#     6: ([8,5,8], [0, 0, 0]),
#     9: ([6,3,6], [0, 0, 0]),
#     12: ([0,5,8], [0, 0, 0]),
#     15: ([2,3,6], [0, 0, 0]),
#     18: ([8,5,8], [0, 0, 0]),
#     21: ([6,3,6], [0, 0, 0]),
#     24: ([0,5,8], [0, 0, 0]),
#     27: ([2,3,6], [0, 0, 0]),
#     30: ([8,5,8], [0, 0, 0]),
#     33: ([6,3,6], [0, 0, 0]),
#     36: ([0,5,8], [0, 0, 0]),
#     39: ([2,3,6], [0, 0, 0]),
#     42: ([8,5,8], [0, 0, 0]),
#     45: ([6,3,6], [0, 0, 0]),
#     48: ([0,5,8], [0, 0, 0]),
#     51: ([6,3,6], [0, 0, 0]),
#     54: ([0,5,8], [0, 0, 0]),
#     57: ([2,3,6], [0, 0, 0]),
#     60: ([8,5,8], [0, 0, 0]),
#     63: ([6,3,6], [0, 0, 0]),
#     # 66: ([0,5,8], [0, 0, 0]),
#     # 69: ([6,3,6], [0, 0, 0]),
#     # 72: ([0,5,8], [0, 0, 0]),
#     # 75: ([2,3,6], [0, 0, 0]),
#     # 78: ([8,5,8], [0, 0, 0]),
#     # 81: ([6,3,6], [0, 0, 0]),
#     # 84: ([0,5,8], [0, 0, 0]),
#     # 87: ([6,3,6], [0, 0, 0]),
#     # 90: ([0,5,8], [0, 0, 0]),
#     # 93: ([2,3,6], [0, 0, 0]),
#     # 96: ([8,5,8], [0, 0, 0]),
#     # 99: ([6,3,6], [0, 0, 0]),
#     # 102: ([0,5,8], [0, 0, 0]),

# }

# p_d_arr = {
#     # 0: (p0_m, [0, 0, 0]),
#     0: ([0,0,8], [0, 0, 0]),
# }


########################################
#             Simulation               #
########################################

t_max = 5
dt = 0.001

imu_misalignment = [1,0,0,0]

accel_bias = [0] * 3
accel_std = [0.02] * 3

gyro_bias = [0] * 3
gyro_std = [0.002] * 3

lidar_bias = [0] * 3
lidar_std = [0.03] * 3

drone_full_navigation = False
drone_use_simple_path = False

filename = "long_data_absolute_state"

DEBUG = True
# DEBUG = False

debug_start_time = 0
# debug_start_time = 7.95   # Seconds into sim to start
# debug_start_time = 11.9
speed_interval = 25  # Frames to travel at once for 0.001 FAST
# speed_interval = 15    # Frames to travel at once for 0.001 SLOW
# speed_interval = 7    # Frames to travel at once for 0.001 SLOW
# speed_interval = 2    # Frames to travel at once for 0.01

########################################
#         Controller Gains             #
########################################

# Attitude

# attitude_controller_1_kp = 3 * [3.0] # GOOD and somewhat related to last row of allocation matrix for kd*r
# attitude_controller_1_kd = 3 * [0.085] # GOOD

# attitude_controller_1_kp = 2 * [100] + [150] # Shin code
attitude_controller_1_kd = 2 * [10] + [10] 
# attitude_controller_1_kd = [0,0,0]

# attitude_controller_1_kp = 2 * [200] + [200] # Shin code
# attitude_controller_1_Lambda = 3 * [5]

attitude_controller_1_kp = 2 * [200] + [200] # Shin code
attitude_controller_1_Lambda = 3 * [5]

# Position

position_controller_1_kp = 2 * [9.5] + [17] #+ [200]
position_controller_1_kd = 2 * [2.4] + [2.7] #+ [75]

# position_controller_1_kp = 3 * [100] #+ [200]
# position_controller_1_kd = 3 * [1] #+ [75]




