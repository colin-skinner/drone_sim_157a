from __future__ import annotations  # For forward references in type hints (Python 3.7+)
import numpy as np
from .quaternion_helpers import *
from typing import Callable


def newtons(f: function, f_dot: function, x: float):
    return x - f(x) / f_dot(x)


def euler_func(
    t: float,
    dt: float,
    x_prev: np.ndarray,
    x_dot: Callable[[float, np.ndarray], np.ndarray],
):
    """Euler's Method using a function for the derivative"""
    x_n = dt * x_dot(t, x_prev) + x_prev

    return x_n



def rk4_func(
    t: float, dt: float, x_prev: float, x_dot: Callable[[float, np.ndarray], np.ndarray]
):
    k1 = x_dot(t, x_prev)
    k2 = x_dot(t + dt / 2, x_prev + dt * k1 / 2)
    k3 = x_dot(t + dt / 2, x_prev + dt * k2 / 2)
    k4 = x_dot(t + dt, x_prev + dt * k3)

    x_n = x_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_n


def rk4_func_interp(
    t: float, dt: float, x_prev: float, x_dot: Callable[[float, np.ndarray], np.ndarray]
):
    k1 = x_dot(t, x_prev)
    k2 = x_dot(t + dt / 2, x_prev + dt * k1 / 2)
    k3 = x_dot(t + dt / 2, x_prev + dt * k2 / 2)
    k4 = x_dot(t + dt, x_prev + dt * k3)

    x_n = x_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_n