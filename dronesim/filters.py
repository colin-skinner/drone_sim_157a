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
        self.F: np.ndarray = None # State transition matrix (to propogate state)
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

    def calc_F(self, w_body: np.ndarray):

        q = self.state[6:10]
        w = w_body
        dt = self.dt

        self.F: np.ndarray = np.eye(10)

        # Position row
        self.F[0:3, 3:6] = np.eye(3) * self.dt

        # Quaternion row
        self.F[6:10, 6:10] = np.array([
            [1, -0.5*dt*w[0], -0.5*dt*w[1], -0.5*dt*w[2]],
            [0.5*dt*w[0], 1, 0.5*dt*w[2], -0.5*dt*w[1]],
            [0.5*dt*w[1], -0.5*dt*w[2], 1, 0.5*dt*w[0]],
            [0.5*dt*w[2], 0.5*dt*w[1], -0.5*dt*w[0], 1]
        ])
        # Angular velocity propogated by measurement

        return self.F

    def predict(self, a_meas_b: np.ndarray, w_meas_body: np.ndarray):
        if any(x is None for x in [self.accel_bias, self.gyro_bias, self.lidar_bias]):
            raise RuntimeError("Call EKF.add_biases()")
        self.a_body = a_meas_b - self.accel_bias
        self.a_body[2] -= 9.81
        self.w_body = w_meas_body - self.gyro_bias

        p = self.state[0:3]
        v = self.state[3:6]
        q = self.state[6:10]
        # w_body = self.state[10:13]


        # assert np.isclose(q, np.ones(4))

        self.a_global = quat_apply(q, self.a_body)

        

        # p
        # print("a_global:", self.a_global)
        # print("computed pos:", p + v * self.dt + 0.5 * self.a_global * self.dt**2)
        self.state[0:3] = p + (v * self.dt) + (0.5 * self.a_global * self.dt * self.dt)
        # print("self.state[0:3] after assignment:", self.state[0:3])
        # breakpoint()

        # v
        self.state[3:6] = v + (self.a_global * self.dt)

        # q
        q_new = q + 0.5 * quat_mult(q, [0, *self.w_body]) * self.dt
        self.state[6:10] = unit(q_new)


        # Propogating P
        self.calc_F(self.w_body)
        self.P = self.F @ self.P @ self.F.T + self.Q

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
        



        





