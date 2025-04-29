import numpy as np
from dronesim import quat_from_axis_rot, ThrustData


########################################
#           Initial State              #
########################################

p0_m = [0, 0, 200]
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

t_max = 100
dt = 0.01

imu_misalignment = [1,0,0,0]

accel_bias = [0,0,0]
accel_std = [0,0,0]

gyro_bias = [0,0,0]
gyro_std = [0,0,0]

lidar_bias = [0,0,0]
lidar_std = [0,0,0]

########################################
#         Controller Gains             #
########################################

attitude_controller_1_kp = 0.03
attitude_controller_1_kd = -0.004

position_controller_1_kp = 3 * [0.1]
position_controller_1_kd = 3 * [-0.6]

# position_controller_1_kp = [2.5, 2.5, 8.0]
# position_controller_1_kd = [1.5, 1.5, 4.5]
