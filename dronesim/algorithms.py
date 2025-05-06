import numpy as np
from .quaternion_helpers import *

class EKF:
    def __init__(self, state0: np.ndarray, P_cov_0: np.ndarray, dt: float):
        assert np.shape(state0) == (10,)
        assert np.shape(P_cov_0) == (10,10)

        self.dt = dt
        
        

        # Initial Nones
        self.accel_bias: np.ndarray = None
        self.gyro_bias: np.ndarray = None
        self.lidar_bias: np.ndarray = None

        # Prediction step matrices
        self.state = state0.copy()
        self.F_dot: np.ndarray = None # State transition matrix (to propogate state)
        self.Q: np.ndarray = None # Process noise covariance (uncertainty in propogation step)

        # Measurements
        self.H: np.ndarray = None # Observation model (maps state to measurements)
        self.R: np.ndarray = np.eye(3) * 0.1**2  # Measurement noise covariance (uncertainty in measurements)

        # Propogation
        self.P = P_cov_0.copy()
        self.P_predict = np.zeros_like(P_cov_0) # For debugging
        self.S: np.ndarray = None # Innovation covariance
        self.K: np.ndarray = None # Kalman gain

        # Process noise
        self.Q = np.diag([ 
            0.1, 0.1, 0.1, # p
            0.1, 0.1, 0.1, # v
            1e-6, 1e-6, 1e-6, 1e-6 # q
        ])

        # self.Q = np.diag([ 
        #     0.01, 0.01, 0.01, # p
        #     0.1, 0.1, 0.1, # v
        #     1e-2, 1e-2, 1e-2, 1e-2 # q
        # ])
        # Lidar noise
        self.y_resid = np.zeros(3)

    def add_biases(self, accel_bias: np.ndarray, gyro_bias: np.ndarray, lidar_bias: np.ndarray):
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias
        self.lidar_bias = lidar_bias

        

    def calc_F_jacobian(self, a_global: np.ndarray, w_body: np.ndarray):

        q = self.state[6:10]
        w = w_body
        dt = self.dt

        self.F_dot: np.ndarray = np.eye(10)

        # Position row
        self.F_dot[0:3, 3:6] = np.eye(3)

        # Quaternion row
        # self.F_dot[6:10, 6:10] = np.array([
        #     [1, -0.5*dt*w[0], -0.5*dt*w[1], -0.5*dt*w[2]],
        #     [0.5*dt*w[0], 1, 0.5*dt*w[2], -0.5*dt*w[1]],
        #     [0.5*dt*w[1], -0.5*dt*w[2], 1, 0.5*dt*w[0]],
        #     [0.5*dt*w[2], 0.5*dt*w[1], -0.5*dt*w[0], 1]
        # ])

        self.F_dot[6:10, 6:10] = 0.5 * np.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0]
        ])


        Q_F = np.array([
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]],
        ])

        # print(a_global)
        term2: np.ndarray = np.zeros((4,4))
        term2[0, 1:4] = a_global
        term2[1:4, 0] = a_global


        term2[1:4, 1:4] = np.array([
            [0, -a_global[2], a_global[1]],
            [a_global[2], 0, -a_global[0]],
            [-a_global[1], a_global[0], 0]
        ])

        # print(term2)

        self.F_dot[3:6, 6:10] = 2 * Q_F @ term2
        # breakpoint()
        # Angular velocity propogated by measurement

        return self.F_dot
    

    def predict(self, a_meas_b: np.ndarray, w_meas_body: np.ndarray):
        """TODO: Add jacobian of H matrix"""
        if any(x is None for x in [self.accel_bias, self.gyro_bias, self.lidar_bias]):
            raise RuntimeError("Call EKF.add_biases()")
        p = self.state[0:3]
        v = self.state[3:6]
        q = unit(self.state[6:10])

        self.a_body = a_meas_b - self.accel_bias
        self.a_global = quat_apply(q, self.a_body)
        self.a_global[2] -= 9.81

        self.w_body = w_meas_body - self.gyro_bias
        self.w_global = quat_apply(q, self.w_body)

        # print(self.a_global)
        # breakpoint()
        

        #### State Transition ####
        self.state[0:3] = p + (v * self.dt) + (0.5 * self.a_global * self.dt * self.dt)

        # v
        self.state[3:6] = v + (self.a_global * self.dt)

        # q
        q_new = q + 0.5 * quat_mult(q, [0, *self.w_body]) * self.dt
        self.state[6:10] = unit(q_new)

        # print(self.state)

        # return

        # TODO: add process noise vector?


        # Propogating P
        self.calc_F_jacobian(self.a_global, self.w_body)
        self.P = self.F_dot @ self.P @ self.F_dot.T + self.Q

    def update(self, p_glob: np.ndarray):

        self.H = np.zeros((3, 10))
        self.H[0:3, 0:3] = np.eye(3)

        # Innovation (basically error)
        self.y_innovation = p_glob - self.state[0:3]


        # Innovation covariance
        self.S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)

        # Update State
        # breakpoint()
        self.state = self.state + self.K @ self.y_innovation
        self.state[6:10] = unit(self.state[6:10])

        # Update covariance
        self.P = (np.eye(10) - self.K @ self.H) @ self.P

        # Update residual (post-update)
        self.y_resid = p_glob - self.state[0:3]






        





