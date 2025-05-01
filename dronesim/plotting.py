import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from time import time, sleep

from .Logger import Logger
from .constants import *
from .quaternion_helpers import quat_apply, norm, unit

# plt.style.use('dark_background')

########################################
#          Vector Helper               #
########################################

def plot_vec_3d(ax: Axes3D, p1: np.array, p2: np.array, color: str = None):
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    zs = [p1[2], p2[2]]

    # print(f"{xs} {ys} {zs}")
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Panel
    if color is None:
        ax.plot(xs, ys, zs)
    else:
        ax.plot(xs, ys, zs, color=color)

########################################
#           1, 2, 3 columns            #
########################################

def plot_3(t: np.ndarray, thing, title=None, factor = 1, fmt = ''):
    f = plt.figure()
    thing = np.array(thing)

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

########################################
#            State Vectors             #
########################################

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

########################################
#            3D             #
########################################

def plot_3d(logger: Logger, max_step = None, title = 'Trajectory', figsize = (20,10), length_unit = 'm', show_fig = True):
    
    fig = plt.figure(figsize = figsize)
    fig.suptitle(title, fontsize = 20)
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    plot_3d_helper(ax, logger, max_step, length_unit)
    fig.show()


def plot_3d_helper(ax: Axes3D, logger: Logger, max_step = None, length_unit = 'm'):

    if max_step is None:
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
        raise RuntimeError("Unrecognized length unit")
    

    x = x_arr[:, 0]
    y = x_arr[:, 1]
    z = x_arr[:, 2]



    x = x_arr[:,0]
    y = x_arr[:,1]
    z = x_arr[:,2]
    

    # ax.set_xlim(-8,8)
    # ax.set_ylim(-8,8)
    # ax.set_zlim(0,5)

    ax.plot(x[:max_step],y[:max_step],z[:max_step])
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


########################################
#            3D with             #
########################################

def debug_3d(logger: Logger,
             title = 'Trajectory Debug',
             figsize = (20,10),
             length_unit = 'm',
             start_time_s = 0, interval = 1):

    max_step = logger.step

    plt.ion()

    fig = plt.figure(figsize = figsize)
    fig.suptitle(title, fontsize = 20)
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    

    start_step = int(start_time_s / logger.sim.dt) + 1
    for step in range(start_step,max_step, interval):

        p = logger.actual_states[:step, 0:3]
        curr_p = p[-1]
        q = logger.actual_states[:step, 6:10]
        curr_q = q[-1]

        torque = logger.actual_torques[step]
        q_d = logger.drone_desired_quat[step]
        p_d_err = logger.drone_p_d_error[step]

        thrust = logger.drone_commanded_force[step]



        plot_3d_helper(ax, logger, step, length_unit)
        ax.text2D(0.05, 0.95, f"Thrust: {norm(thrust)}", transform=ax.transAxes)

        
        # Coordinate system
        # plot_vec_3d(ax, curr_p, curr_p + unit(torque), 'black')
        plot_vec_3d(ax, curr_p, curr_p + quat_apply(curr_q, [1,0,0]), 'red')
        plot_vec_3d(ax, curr_p, curr_p + quat_apply(curr_q, [0,1,0]), 'green')
        plot_vec_3d(ax, curr_p, curr_p + quat_apply(curr_q, [0,0,1]), 'blue')

        plot_vec_3d(ax, curr_p, curr_p + unit(quat_apply(q_d, [0.5,0,0])), 'purple')
        plot_vec_3d(ax, curr_p, curr_p + unit(quat_apply(q_d, [0,0.5,0])), 'orange')
        plot_vec_3d(ax, curr_p, curr_p + unit(quat_apply(q_d, [0,0,0.5])), 'black')

        plot_vec_3d(ax, curr_p, curr_p + p_d_err, 'brown')
        plot_vec_3d(ax, curr_p, curr_p + thrust, 'gray')
        # fig.show()

        ax.set_xlim(min(logger.actual_states[:max_step, 0] - 0.5), max(logger.actual_states[:max_step, 0]) + 0.5)
        ax.set_ylim(min(logger.actual_states[:max_step, 1] - 0.5), max(logger.actual_states[:max_step, 1]) + 0.5)
        ax.set_zlim(min(logger.actual_states[:max_step, 2] - 0.5), max(logger.actual_states[:max_step, 2]) + 0.5)
        ax.set_title(f"{title}: {logger.t[step]}s")

        ax.legend(["Trajectory", "x_axis", "y_axis", "z_axis", "x_d", "y_d", "z_d", "p_err"])
        # ax.set_aspect('equal', adjustable='box')
        # ax.axis('square')
        plt.pause(logger.sim.dt)

        if step + interval + 1 > max_step:
            if input("Press w to watch again:") == "w":
                debug_3d(logger, title, figsize, length_unit, start_time_s, interval)


        ax.cla()

        



########################################
#            State Vectors             #
########################################