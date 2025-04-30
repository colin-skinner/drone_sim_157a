import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .Logger import Logger
from .constants import *
from .quaternion_helpers import quat_apply

# plt.style.use('dark_background')



def plot_3(t: np.ndarray, thing, title=None, factor = 1, fmt = ''):
    f = plt.figure()
    thing = np.array(thing)
    # print(np.shape(t))
    # print(np.shape(thing))

    # print(t[-4:])
    # print(thing[-4:])
    # breakpoint()
    plt.plot(t, thing[:,0] * factor, fmt, label = "X")
    plt.plot(t, thing[:,1] * factor, fmt, label = "Y")
    plt.plot(t, thing[:,2] * factor, fmt, label = "Z")
    # plt.xlim(0,100)
    # plt.ylim(0,100)
    # plt.zlim(0,100)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    f.show()

def plot_2(t, thing, title=None, factor = 1):
    f = plt.figure()
    plt.plot(t, thing * factor, label = "x")
    plt.plot(t, thing * factor, label = "y")
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    f.show()

def plot_1(t, thing, title=None, factor = 1):
    f = plt.figure()
    plt.plot(t, thing * factor)
    if title:
        plt.title(title)
    plt.grid(True)
    f.show()


def plot_drone_axis(logger: Logger, axis: list | np.ndarray, title=None):
    
    axis_vec = np.array([
        quat_apply(q_B2L, axis) for q_B2L in logger.actual_states[0:logger.step, 6:10].tolist()
    ])
    plot_3(logger.t[0:logger.step], axis_vec, title)


def plot_state_vector(
    logger: Logger,
    figsize=(20, 9.5),
    time_unit="s",
    length_unit="m",
    *,
    title: str = None,
):

    max_step = logger.step
    t = logger.t[:max_step]


    # if time_unit == "minute":
    #     t = t * SEC2MIN
    # elif time_unit == "hour":
    #     t = t * SEC2HOUR
    # elif time_unit == "day":
    #     t = t * SEC2DAY
    # elif time_unit in ["second", "s"]:
    #     t = t
    # else:
    #     print("Unrecognized time unit")
    #     return

    p = logger.actual_states[:max_step, 0:3]
    v = logger.actual_states[:max_step, 3:6]
    q = logger.actual_states[:max_step, 6:10]
    w = logger.actual_states[:max_step, 10:13]

    if length_unit in ["meter", "m"]:
        x_arr = p
        v = v
    elif length_unit in ["centimeter", "cm"]:
        x_arr = p * M2CM
        v = v * M2CM
    elif length_unit in ["foot", "ft"]:
        x_arr = p * M2FT
        v = v * M2FT
    else:
        print("Unrecognized length unit")
        return



    x = x_arr[:, 0]
    y = x_arr[:, 1]
    z = x_arr[:, 2]

    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]



    ########################################
    #             Plotting                 #
    ########################################

    figure, axs = plt.subplots(nrows=2, ncols=3, figsize=figsize)

    if title is None:
        title = "Drone States (position, velocity)"

    figure.suptitle(title, fontsize=20)

    fig: Axes3D = axs[0, 0]
    fig.plot(t, x)
    fig.set_title("X Position vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"X ({length_unit})")
    fig.set_xlabel(f"time ({time_unit})")

    fig: Axes3D = axs[0, 1]
    fig.plot(t, y)
    fig.set_title("Y Position vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"Y ({length_unit})")
    fig.set_xlabel(f"time ({time_unit})")

    fig: Axes3D = axs[0, 2]
    fig.plot(t, z)
    fig.set_title("Z Position vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"Z ({length_unit})")
    fig.set_xlabel(f"time ({time_unit})")

    fig: Axes3D = axs[1, 0]
    fig.plot(t, vx)
    fig.set_title("X Velocity vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"Vx ({length_unit}/{time_unit})")
    fig.set_xlabel(f"time ({time_unit})")

    fig: Axes3D = axs[1, 1]
    fig.plot(t, vy)
    fig.set_title("Y Velocity vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"Vy ({length_unit}/{time_unit})")
    fig.set_xlabel(f"time ({time_unit})")

    fig: Axes3D = axs[1, 2]
    fig.plot(t, vz)
    fig.set_title("Z Velocity vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"Vz ({length_unit}/{time_unit})")
    fig.set_xlabel(f"time ({time_unit})")

    # Angular velocity
    figure, axs = plt.subplots(nrows=1, ncols=3, figsize=(figsize[0], figsize[1] / 2))

    if title is None:
        title = "Drone States (angular velocity)"

    figure.suptitle(title, fontsize=20)

    fig: Axes3D = axs[0]
    fig.plot(t, w[:,0])
    fig.set_title("X Angular Velocity vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"wX ({time_unit})")
    fig.set_xlabel(f"time ({time_unit})")

    fig: Axes3D = axs[1]
    fig.plot(t, w[:,1])
    fig.set_title("Y Angular Velocity vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"wY (rad/{time_unit})")
    fig.set_xlabel(f"time ({time_unit})")

    fig: Axes3D = axs[2]
    fig.plot(t, w[:,2])
    fig.set_title("Z Angular Velocity vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"wZ (rad/{time_unit})")
    fig.set_xlabel(f"time ({time_unit})")

    figure.show()


def plot_3d(logger: Logger, title = 'Trajectory', figsize = (20,10), time_unit = 'second', length_unit = 'meter'):


    fig = plt.figure(figsize = figsize)

    fig.suptitle(title, fontsize = 20)

    max_step = logger.step

    p = logger.actual_states[:max_step, 0:3]
    v = logger.actual_states[:max_step, 3:6]
    q = logger.actual_states[:max_step, 6:10]
    w = logger.actual_states[:max_step, 10:13]

    if length_unit in ["meter", "m"]:
        x_arr = p
        v = v
    elif length_unit in ["centimeter", "cm"]:
        x_arr = p * M2CM
        v = v * M2CM
    elif length_unit in ["foot", "ft"]:
        x_arr = p * M2FT
        v = v * M2FT
    else:
        print("Unrecognized length unit")
        return
    

    x = x_arr[:, 0]
    y = x_arr[:, 1]
    z = x_arr[:, 2]



    x = x_arr[:,0]
    y = x_arr[:,1]
    z = x_arr[:,2]
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal', adjustable='box')
    # ax.axis('square')

    # ax.set_zlim(self.initial_altitude, max(x_arr[:,2]))

    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)
    ax.set_zlim(0,5)

    ax.plot(x[:max_step],y[:max_step],z[:max_step])
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.show()