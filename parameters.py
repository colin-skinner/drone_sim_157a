import numpy as np
from dronesim import quat_from_axis_rot, ThrustData


########################################
#           Initial State              #
########################################

p0_m = [0, 0, 2]
v0_m = [0, 0, 0]
q0 = [1, 0, 0, 0]  # Identity quaternion
# q0 = quat_from_axis_rot(100, [1, 1, 0]).tolist()  # 20 deg angle in y
w0_rad_s = [0, 0, 0]
state0 = np.array(p0_m + v0_m + q0 + w0_rad_s)

########################################
#             Mass Stuff               #
########################################

mass = 1  # kg
I = np.array([[0.00030, 0, 0], [0, 0.00030, 0], [0, 0, -0.00045]])
dimensions = np.array([13, 13, 8])  # input into list as cm


########################################
#            Prop Stuff                #
########################################

thrust_data = ThrustData("Calibration Data/Motor_Kv1860_Orange_Propeller_Data.xlsx", drop_duplicates=True)
print(thrust_data.lookup_table)

min_prop_force_kgf = min(thrust_data.lookup_table["Thrust (kgf)"])
max_prop_force_kgf = max(thrust_data.lookup_table["Thrust (kgf)"])

# min_prop_force_kgf = 0.095
# max_prop_force_kgf = 0.46

print(f"{min_prop_force_kgf=}")
print(f"{max_prop_force_kgf=}")

# ADDD LOOKUP TABLE PROP

########################################
#             Simulation               #
########################################

t_max = 20
dt = 0.01

imu_misalignment = [1,0,0,0]

accel_bias = [0,0,0]
accel_std = [0,0,0]

gyro_bias = [0,0,0]
gyro_std = [0,0,0]

lidar_bias = [0,0,0]
lidar_std = [0,0,0]

########################################
#               Path                   #
########################################

p_d_arr = { # testing Z
    0: [0,0,3],
    2: [0,0,4],
    4: [0,0,5],
    6: [0,0,4],
    8: [0,0,3],
}

p_d_arr = { # testing X
    0: [0,0,2],
    2: [1,0,2],
    4: [2,0,2],
    6: [3,0,2],
    8: [4,0,2],
}


# p_d_arr = { # testing X
#     0: [0,0,2],
#     2: [0,1,2],
#     4: [0,2,2],
#     6: [0,3,2],
#     8: [0,4,2],
# }

# p_d_arr = { # weird one
#     0: [0,0,2],
#     2: [1,0,2],
#     4: [1,0.1,2],
#     # 6: [0,3,2],
#     # 8: [0,4,2],
# }

########################################
#         Controller Gains             #
########################################

# attitude_controller_1_kp = 0.3 # GOOD
# attitude_controller_1_kd = -0.004 # GOOD

attitude_controller_1_kp = 0.3
attitude_controller_1_kd = -0.004
# attitude_controller_1_kd = -0.2 * np.sqrt(attitude_controller_1_kp)

# position_controller_1_kp = 3 * [0.1]
# position_controller_1_kd = 3 * [-0.6]

# a = 3
# b = 0.4
a = 10.5
b = 5.5
position_controller_1_kp = [10.5,a, 10.5] # good Z
position_controller_1_kd = [5.5,b, 5.5] # good Z



