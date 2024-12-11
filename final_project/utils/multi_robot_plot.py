"""
Plotting tool for 2D multi-robot system

author: Ashwin Bose (@atb033)
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np


def plot_robots_and_obstacles(robots, obstacles, robot_radius, num_steps, sim_time, filename):
    """
    绘制多个机器人和障碍物的动画。

    :param robots: list，每个元素是机器人轨迹的 numpy 数组，形状为 (2, num_steps)
    :param obstacles: numpy 数组，形状为 (2, num_steps, num_obstacles)
    :param robot_radius: float，机器人的半径
    :param num_steps: int，仿真的时间步数
    :param sim_time: float，总仿真时间
    :param filename: str，保存的文件名
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect('equal')
    ax.grid()

    # 机器人和轨迹初始化
    robot_patches = []
    lines = []
    for robot in robots:
        patch = Circle((robot[0, 0], robot[1, 0]), robot_radius, facecolor='green', edgecolor='black')
        line, = ax.plot([], [], '--r')
        robot_patches.append(patch)
        lines.append(line)

    # 障碍物初始化
    obstacle_patches = []
    for obstacle in range(obstacles.shape[2]):
        obstacle_patch = Circle((0, 0), robot_radius, facecolor='aqua', edgecolor='black')
        obstacle_patches.append(obstacle_patch)

    def init():
        # 添加机器人和障碍物到图形
        for patch in robot_patches:
            ax.add_patch(patch)
        for obstacle_patch in obstacle_patches:
            ax.add_patch(obstacle_patch)
        for line in lines:
            line.set_data([], [])
        return robot_patches + lines + obstacle_patches

    def animate(i):
        # 更新每个机器人的位置和轨迹
        for idx, robot in enumerate(robots):
            robot_patches[idx].center = (robot[0, i], robot[1, i])
            lines[idx].set_data(robot[0, :i], robot[1, :i])

        # 更新每个障碍物的位置
        for j in range(len(obstacle_patches)):
            obstacle_patches[j].center = (obstacles[0, i, j], obstacles[1, i, j])

        return robot_patches + lines + obstacle_patches

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


def plot_robot_and_obstacles(robot, obstacles, robot_radius, num_steps, sim_time, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect('equal')
    ax.grid()
    line, = ax.plot([], [], '--r')

    robot_patch = Circle((robot[0, 0], robot[1, 0]),
                         robot_radius, facecolor='green', edgecolor='black')
    obstacle_list = []
    for obstacle in range(np.shape(obstacles)[2]):
        obstacle = Circle((0, 0), robot_radius,
                          facecolor='aqua', edgecolor='black')
        obstacle_list.append(obstacle)

    def init():
        ax.add_patch(robot_patch)
        for obstacle in obstacle_list:
            ax.add_patch(obstacle)
        line.set_data([], [])
        return [robot_patch] + [line] + obstacle_list

    def animate(i):
        robot_patch.center = (robot[0, i], robot[1, i])
        for j in range(len(obstacle_list)):
            obstacle_list[j].center = (obstacles[0, i, j], obstacles[1, i, j])
        line.set_data(robot[0, :i], robot[1, :i])
        return [robot_patch] + [line] + obstacle_list

    init()
    step = (sim_time / num_steps)
    for i in range(num_steps):
        animate(i)
        plt.pause(step)

    # Save animation
    if not filename:
        return

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, num_steps), interval=200,
        blit=True, init_func=init)

    ani.save(filename, "ffmpeg", fps=30)


def plot_robot(robot, timestep, radius=1, is_obstacle=False):
    if robot is None:
        return
    center = robot[:2, timestep]
    x = center[0]
    y = center[1]
    if is_obstacle:
        circle = plt.Circle((x, y), radius, color='aqua', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], '--r',)
    else:
        circle = plt.Circle((x, y), radius, color='green', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], 'blue')

    plt.gcf().gca().add_artist(circle)
