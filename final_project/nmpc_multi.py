import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import animation
import yaml
from concurrent.futures import ThreadPoolExecutor


# Global parameters
SIM_TIME = 30.0  # Total simulation time
TIMESTEP = 0.1  # Time interval for each step
NUMBER_OF_TIMESTEPS = int(SIM_TIME / TIMESTEP)
ROBOT_RADIUS = 0.2  # Radius of the robot
obstacle_radius = 0.5  # Radius of obstacles
VMAX = 2  # Maximum velocity
VMIN = 0.2  # Minimum velocity

# Collision cost parameters
Qc = 3.0  # Weight for collision cost
kappa = 4.0  # Adjustment coefficient for collision cost

# NMPC parameters
HORIZON_LENGTH = 4  # Prediction horizon length
NMPC_TIMESTEP = 0.3  # NMPC computation timestep
upper_bound = [(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2

# Fixed obstacles
with open("env_2.yaml", 'r') as param_file:
    try:
        param = yaml.load(param_file, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)


def generate_boundary_obstacles(length, width):
    """
    Generate boundary obstacles around a 10x10 grid.

    :param length: Length of the grid (10)
    :param width: Width of the grid (10)
    :return: Numpy array of obstacle coordinates, shape (n, 2)
    """
    obstacles = []
    obstacles_inside = param["map"]["obstacles"]
    # Bottom and top edges
    for i in range(length):
        obstacles.append([i, -1])  # Bottom edge
        obstacles.append([i, width])  # Top edge

    # Left and right edges
    for i in range(width):
        obstacles.append([-1, i])  # Left edge
        obstacles.append([length, i])  # Right edge

    # Corners
    obstacles.append([-1, -1])  # Bottom-left corner
    obstacles.append([length, -1])  # Bottom-right corner
    obstacles.append([-1, width])  # Top-left corner
    obstacles.append([length, width])  # Top-right corner

    return obstacles


dimension = param["map"]["dimensions"]
obstacles = param["map"]["obstacles"] + generate_boundary_obstacles(dimension[0], dimension[1])
agents = np.array(param['agents'])


# Main function: Simulate path planning
def simulate(filename):
    # Initialize starting and goal positions for two robots
    robot_start_positions = [np.array(agents[0]['start']), np.array(agents[1]['start'])]
    robot_goals = [np.array(agents[0]['goal']), np.array(agents[1]['goal'])]

    # Record robot states over time
    robot_states = robot_start_positions.copy()
    robot_histories = [np.empty((2, NUMBER_OF_TIMESTEPS)) for _ in range(2)]

    # Create a thread pool
    with ThreadPoolExecutor(max_workers=2) as executor:
        for t in range(NUMBER_OF_TIMESTEPS):
            # Create task list
            tasks = [
                (robot_states[i], obstacles, compute_xref(robot_states[i], robot_goals[i], HORIZON_LENGTH, NMPC_TIMESTEP))
                for i in range(len(robot_states))
            ]

            # Compute velocities for all robots in parallel
            results = list(executor.map(lambda args: compute_velocity(*args), tasks))

            # Update robot states
            for i, (vel, _) in enumerate(results):
                robot_states[i] = update_state(robot_states[i], vel, TIMESTEP)
                robot_histories[i][:, t] = robot_states[i]

    # Plot robots and obstacles
    plot_robots_and_obstacles(robot_histories, obstacles, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)


# Compute optimal control input
def compute_velocity(robot_state, fixed_obstacles, xref):
    filtered_obstacles = filter_obstacles(robot_state, fixed_obstacles)

    u0 = np.random.rand(2 * HORIZON_LENGTH)  # Initialize random control sequence

    def cost_fn(u):
        return total_cost(u, robot_state, filtered_obstacles, xref)

    bounds = Bounds(lower_bound, upper_bound)
    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x


# Compute reference trajectory
def compute_xref(start, goal, number_of_steps, timestep):
    dir_vec = (goal - start).astype(np.float64)  # Ensure floating-point computation
    norm = np.linalg.norm(dir_vec)
    if norm < 0.1:
        new_goal = start
    else:
        dir_vec /= norm  # Normalize direction vector
        new_goal = start + dir_vec * VMAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).flatten()


# Total cost function
def total_cost(u, robot_state, fixed_obstacles, xref):
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)  # Trajectory tracking cost
    c2 = total_collision_cost(x_robot, fixed_obstacles)  # Collision cost
    return c1 + c2


# Trajectory tracking cost
def tracking_cost(x, xref):
    return np.linalg.norm(x - xref)


# Total collision cost
def total_collision_cost(robot, obstacles):
    total_cost = 0
    for i in range(HORIZON_LENGTH):
        rob = robot[2 * i: 2 * i + 2]
        for obs in obstacles:
            total_cost += collision_cost(rob, obs)
    return total_cost


# Single-step collision cost
def collision_cost(x0, x1):
    effective_radius = ROBOT_RADIUS + obstacle_radius
    d = np.linalg.norm(x0 - x1)
    return Qc / (1 + np.exp(kappa * (d - effective_radius)))


# Update robot state
def update_state(x0, u, timestep):
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))
    new_state = np.vstack([np.eye(2)] * N) @ x0 + kron @ u * timestep
    return new_state


def filter_obstacles(robot_state, obstacles, radius=5.0):
    # Only retain obstacles within a certain radius
    filtered = [obs for obs in obstacles if np.linalg.norm(robot_state - obs) < radius]
    return np.array(filtered)


# Plot robot and obstacle trajectories
def plot_robots_and_obstacles(robots, obstacles, robot_radius, num_steps, sim_time, filename):
    """
    Plot the movement of multiple robots and obstacles as an animation, and save to a file.

    :param robots: list, each element is a numpy array of robot trajectories, shape (2, num_steps)
    :param obstacles: numpy array, shape (n_obstacles, 2), positions of fixed obstacles
    :param robot_radius: float, radius of the robot
    :param num_steps: int, number of simulation steps
    :param sim_time: float, total simulation time
    :param filename: str, name of the file to save
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 10.5)
    ax.set_ylim(-1.5, 10.5)
    ax.set_aspect('equal')
    ax.grid()

    # Draw fixed obstacles
    for obs in obstacles:
        ax.add_patch(Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, edgecolor="black", alpha=1))

    # Initialize robots and trajectories
    robot_patches = []
    trajectory_lines = []  # Store trajectory line objects for each robot
    for robot in robots:
        # Initialize robot
        patch = Circle((robot[0, 0], robot[1, 0]), robot_radius, facecolor='green', edgecolor='black')
        robot_patches.append(patch)
        ax.add_patch(patch)

        # Initialize robot trajectory
        line, = ax.plot([], [], '--', label=f'Robot {len(robot_patches)}')  # Dashed line for trajectory
        trajectory_lines.append(line)

    # Animation initialization
    def init():
        for patch in robot_patches:
            patch.center = (0, 0)  # Initialize robot position
        for line in trajectory_lines:
            line.set_data([], [])  # Clear trajectories
        return robot_patches + trajectory_lines

    # Animation update
    def animate(i):
        for idx, robot in enumerate(robots):
            # Update robot position
            robot_patches[idx].center = (robot[0, i], robot[1, i])
            # Update trajectory
            trajectory_lines[idx].set_data(robot[0, :i + 1], robot[1, :i + 1])  # Accumulate trajectory
        return robot_patches + trajectory_lines

    # Real-time animation display
    init()
    step = sim_time / num_steps
    for i in range(num_steps):
        animate(i)
        plt.pause(step)

    # Save animation
    if filename:
        ani = animation.FuncAnimation(
            fig, animate, frames=np.arange(1, num_steps), interval=200, blit=True, init_func=init)
        ani.save(filename, "ffmpeg", fps=30)


# Run simulation
simulate("multi_robot.mp4")
