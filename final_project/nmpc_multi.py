import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import animation
import yaml

# 全局参数
SIM_TIME = 30.0                # 仿真总时间
TIMESTEP = 0.1                # 单步时间间隔
NUMBER_OF_TIMESTEPS = int(SIM_TIME / TIMESTEP)
ROBOT_RADIUS = 0.2            # 机器人的半径
obstacle_radius = 0.5         # 障碍物的半径
VMAX = 2                      # 最大速度
VMIN = 0.2                    # 最小速度

# 碰撞代价参数
Qc = 5.0                      # 碰撞代价权重
kappa = 4.0                   # 碰撞代价调整系数

# NMPC 参数
HORIZON_LENGTH = 8            # 预测时域长度
NMPC_TIMESTEP = 0.3           # NMPC计算的时间步
upper_bound = [(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2

# 固定障碍物
with open("env_2.yaml", 'r') as param_file:
    try:
        param = yaml.load(param_file, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)

def generate_boundary_obstacles(length, width):
    """
    生成围绕10x10区域的边界障碍物。

    :param length: 网格的长度（10）
    :param width: 网格的宽度（10）
    :return: numpy 数组，障碍物的坐标，形状为 (n, 2)
    """
    obstacles = []
    obstacles_inside = param["map"]["obstacles"]
    # 底边和顶边
    for i in range(length):
        obstacles.append([i, -1])  # 底边
        obstacles.append([i, width])  # 顶边

    # 左边和右边
    for i in range(width):
        obstacles.append([-1, i])  # 左边
        obstacles.append([length, i])  # 右边

    # 四个角的障碍物
    obstacles.append([-1, -1])  # 左下角
    obstacles.append([length, -1])  # 右下角
    obstacles.append([-1, width])  # 左上角
    obstacles.append([length, width])  # 右上角

    return obstacles

dimension = param["map"]["dimensions"]
obstacles = param["map"]["obstacles"] + generate_boundary_obstacles(dimension[0], dimension[1])
agents = np.array(param['agents'])

# 主函数：仿真路径规划
def simulate(filename):
    # 初始化两台机器人的起始位置和目标位置
    robot_start_positions = [np.array(agents[0]['start']), np.array(agents[1]['start'])]
    robot_goals = [np.array(agents[0]['goal']), np.array(agents[1]['goal'])]

    # 记录机器人状态历史
    robot_states = robot_start_positions.copy()
    robot_histories = [np.empty((2, NUMBER_OF_TIMESTEPS)) for _ in range(2)]

    # 仿真每个时间步
    for t in range(NUMBER_OF_TIMESTEPS):
        for robot_index in range(len(robot_states)):
            # 计算当前机器人的参考轨迹
            xref = compute_xref(robot_states[robot_index], robot_goals[robot_index], HORIZON_LENGTH, NMPC_TIMESTEP)

            # 计算速度，考虑固定障碍物
            vel, _ = compute_velocity(robot_states[robot_index], obstacles, xref)

            # 更新机器人状态
            robot_states[robot_index] = update_state(robot_states[robot_index], vel, TIMESTEP)
            robot_histories[robot_index][:, t] = robot_states[robot_index]

    # 绘制机器人和障碍物
    plot_robots_and_obstacles(robot_histories, obstacles, ROBOT_RADIUS,obstacle_radius, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)


# 计算最优控制输入
def compute_velocity(robot_state, fixed_obstacles, xref):
    u0 = np.random.rand(2 * HORIZON_LENGTH)  # 初始化随机控制序列

    def cost_fn(u):
        return total_cost(u, robot_state, fixed_obstacles, xref)

    bounds = Bounds(lower_bound, upper_bound)
    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x


# 计算参考轨迹
def compute_xref(start, goal, number_of_steps, timestep):
    dir_vec = (goal - start).astype(np.float64)  # 确保是浮点数运算
    norm = np.linalg.norm(dir_vec)
    if norm < 0.1:
        new_goal = start
    else:
        dir_vec /= norm  # 单位化方向向量
        new_goal = start + dir_vec * VMAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).flatten()



# 总代价函数
def total_cost(u, robot_state, fixed_obstacles, xref):
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)           # 轨迹跟踪代价
    c2 = total_collision_cost(x_robot, fixed_obstacles)  # 碰撞代价
    return c1 + c2


# 轨迹跟踪代价
def tracking_cost(x, xref):
    return np.linalg.norm(x - xref)


# 碰撞代价
def total_collision_cost(robot, obstacles):
    total_cost = 0
    for i in range(HORIZON_LENGTH):
        rob = robot[2 * i: 2 * i + 2]
        for obs in obstacles:
            total_cost += collision_cost(rob, obs)
    return total_cost


# 单步碰撞代价
def collision_cost(x0, x1):
    effective_radius = ROBOT_RADIUS + obstacle_radius
    d = np.linalg.norm(x0 - x1)
    return Qc / (1 + np.exp(kappa * (d - effective_radius)))


# 更新机器人状态
def update_state(x0, u, timestep):
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))
    new_state = np.vstack([np.eye(2)] * N) @ x0 + kron @ u * timestep
    return new_state


# 绘制机器人和障碍物的运动轨迹
def plot_robots_and_obstacles(robots, obstacles, robot_radius,obstacle_radius, num_steps, sim_time, filename):
    """
    绘制多个机器人和障碍物的动画，并保存到文件。

    :param robots: list，每个元素是机器人轨迹的 numpy 数组，形状为 (2, num_steps)
    :param obstacles: numpy 数组，形状为 (n_obstacles, 2)，固定障碍物的位置
    :param robot_radius: float，机器人的半径
    :param num_steps: int，仿真的时间步数
    :param sim_time: float，总仿真时间
    :param filename: str，保存的文件名
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 10.5)
    ax.set_ylim(-1.5, 10.5)
    ax.set_aspect('equal')
    ax.grid()
    line, = ax.plot([], [], '--r')

    # 绘制固定障碍物
    for obs in obstacles:
        ax.add_patch(Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color="blue",edgecolor="black", alpha=1))


    # 初始化机器人和轨迹
    robot_patches = []
    for robot in robots:
        patch = Circle((robot[0, 0], robot[1, 0]), robot_radius, facecolor='green', edgecolor='black')
        robot_patches.append(patch)
        ax.add_patch(patch)

    # 动画初始化
    def init():
        for patch in robot_patches:
            patch.center = (0, 0)  # 初始化在屏幕外的点
        return robot_patches

    # def init():
    #     # ax.add_patch(robot_patch)
    #     # for obstacle in obstacle_list:
    #     #     ax.add_patch(obstacle)
    #     line.set_data([], [])
    #     return [robot_patch] + [line] + obstacle_list

    # 动画更新
    def animate(i):
        for idx, robot in enumerate(robots):
            robot_patches[idx].center = (robot[0, i], robot[1, i])
        return robot_patches

    # 动画实时显示
    init()
    step = sim_time / num_steps
    for i in range(num_steps):
        animate(i)
        plt.pause(step)

    # 保存动画
    if filename:
        ani = animation.FuncAnimation(
            fig, animate, frames=np.arange(1, num_steps), interval=200, blit=True, init_func=init)
        ani.save(filename, "ffmpeg", fps=30)


# 运行仿真
simulate("multi_robot.mp4")
