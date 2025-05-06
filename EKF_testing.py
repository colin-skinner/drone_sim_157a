import numpy as np
from dronesim import *
from parameters import *
import os, csv
import pandas as pd
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
    
filename = "results/x_test_kalman.csv"

data = pd.read_csv(f'{os.getcwd()}/{filename}')

w_body = np.array(data[["wx_body (rad/s2)", "wy_body (rad/s2)", "wz_body (rad/s2)"]])
a_body = np.array(data[["ax_body (m/s2)", "ay_body (m/s2)", "az_body (m/s2)"]])
x_actual = np.array(data[["x_actual (m)", "y_actual (m)", "z_actual (m)"]])
t = np.array(data["Time (s)"])
dt = t[1] - t[0]

a_body = a_body + np.random.normal(0, 0.02, a_body.shape) 
w_body = w_body + np.random.normal(0, 0.002, w_body.shape) 


# Cov
P0 = np.zeros((10, 10))
P0[0:3, 0:3] = np.eye(3) * 0.05**2                  # p in m
P0[3:6, 3:6] = np.eye(3) * 0.05**2                  # v in m/s
P0[6:10, 6:10] = np.eye(4) * 1e-5                   # q
# P0[10:13, 10:13] = np.eye(3) * (5 * DEG2RAD)**2     # w in rad/s

# EKF testing

ekf = EKF(state0[0:10], P0, dt)

ekf.add_biases(accel_bias, gyro_bias, lidar_bias)

size = len(data)

states = np.zeros((size, 10))
resids = np.zeros((size, 3))

print(ekf.state)
for i in range(size):

    accel = a_body[i]
    gyro = w_body[i]
    lidar = x_actual[i]

    # breakpoint()
    # print(ekf.state)

    ekf.predict(accel, gyro)
    # print(ekf.state)

    # if i % 100 == 1:
    #     ekf.update(lidar)
    #     print("AH")
    # print(ekf.state)

    if i % 100 == 0:
        print(f"{i} out of {size}")

    print(i)

    states[i, :] = ekf.state
    resids[i, :] = ekf.y_resid


plt.figure()
plt.plot(t, x_actual[:,0], label="X")
plt.plot(t, states[:,0], label="X mean")
plt.ylim([-10,10])
plt.ylabel("Position (m)")
plt.xlabel("Time (s)")
plt.legend()

plt.figure()
plt.plot(t, x_actual[:,1], label="Y")
plt.plot(t, states[:,1], label="Y meas")
plt.ylim([-10,10])
plt.ylabel("Position (m)")
plt.xlabel("Time (s)")
plt.legend()

plt.figure()
plt.plot(t, x_actual[:,2], label="Z")
plt.plot(t, states[:,2], label="Z meas")
plt.ylim([-10,10])
plt.ylabel("Position (m)")
plt.xlabel("Time (s)")
plt.legend()

axis_vec = np.array([
    quat_apply(q_B2L, [0,0,1]) for q_B2L in states[:,6:10].tolist()
])
plt.figure()
plt.plot(t, axis_vec, label="axis")

plt.xlabel("Time (s)")
plt.legend()


plt.show()

stds = np.std(resids, axis=0)
print("Residual STD [m]:", stds)

plt.figure()
plt.plot(resids, label=["x", "y", "z"])
plt.legend()
plt.ylabel("Residual (m)")
plt.xlabel("Time step")
plt.show()

plt.figure()
plt.hist(resids, bins=50, label=["x", "y", "z"])
plt.title("Histogram of z residuals")
plt.show()
    