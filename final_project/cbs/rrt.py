import random
import math
from math import fabs
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_rrt_path(tree, path, obstacles=None, start=None, goal=None):
    """
    绘制RRT生成的路径以及障碍物和起点目标点
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制障碍物（如果有的话）
    if obstacles:
        for obs in obstacles:
            ax.add_patch(plt.Rectangle((obs[0], obs[1] - 0.5), 1, 0.5, color="gray", alpha=0.5))

    # 绘制起点和目标点
    if start:
        ax.plot(start.x, start.y, 'go', label='Start')
    if goal:
        ax.plot(goal.x, goal.y, 'ro', label='Goal')

    # 绘制路径
    if path:
        path_x = [state.location.x for state in path]
        path_y = [state.location.y for state in path]
        ax.plot(path_x, path_y, 'b-', label='Path')

    # 绘制树
    tree_x = [state.location.x for state in tree]
    tree_y = [state.location.y for state in tree]
    ax.plot(tree_x, tree_y, 'k.', label='RRT Tree')

    # 设置图形显示参数
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("RRT Path Planning")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # 显示图形
    plt.show()


class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time) + str(self.location))

    def __str__(self):
        return '(' + str(self.time) + ', ' + str(self.location) + ')'


class Location(object):
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return str((self.x, self.y))


class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time) + str(self.location.x) + str(self.location.y))

    def is_equal_except_time(self, state):
        return self.location == state.location

    def is_around(self, state, bias):
        return fabs(self.location.x - state.location.x) < bias and fabs(self.location.y - state.location.y) < bias

    def __str__(self):
        return str((self.time, self.location.x, self.location.y))


class RRT(object):
    def __init__(self, env):
        self.dimension = env.dimension
        self.obstacles = env.obstacles
        self.constraints = env.constraints
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic  # 可选，如果需要可以在RRT中使用启发式函数
        self.is_at_goal = env.is_at_goal
        self.is_around_goal = env.is_around_goal
        self.get_neighbors = env.get_neighbors  # 不一定需要在RRT中使用
        self.max_iterations = 100000  # 最大扩展次数
        self.step_size = 0.1  # 步长
        self.tree = []

    def distance(self, p1, p2):
        """计算两点之间的欧几里得距离"""
        return math.sqrt((p1.location.x - p2.location.x) ** 2 + (p1.location.y - p2.location.y) ** 2)

    def nearest(self, tree, node):
        """寻找树中离目标节点最近的节点"""
        return min(tree, key=lambda n: self.distance(n, node))

    def steer(self, from_node, to_node):
        """从当前节点朝目标节点移动，步长为step_size，扩展时不穿过障碍物"""
        dist = self.distance(from_node, to_node)
        if dist < self.step_size:
            return to_node
        else:
            # 计算单位向量，乘以步长得到新的节点位置
            direction = ((to_node.location.x - from_node.location.x) / dist,
                         (to_node.location.y - from_node.location.y) / dist)
            new_node = State(from_node.time + 1,
                             Location(from_node.location.x + direction[0] * self.step_size,
                                      from_node.location.y + direction[1] * self.step_size))

            # 在扩展过程中检查路径是否与障碍物相交
            if self.line_intersects_obstacle(
                    (from_node.location.x, from_node.location.y),
                    (new_node.location.x, new_node.location.y),
                    self.obstacles):
                return None  # 如果路径穿越了障碍物，返回None表示无效路径

            return new_node

    def search(self, agent_name):
        """
        低级别的RRT搜索
        """
        initial_state = self.agent_dict[agent_name]["start"]
        goal_state = self.agent_dict[agent_name]["goal"]
        max_iterations = self.max_iterations

        self.tree = [initial_state]  # 初始化树，树的根节点是 initial_state
        parent_map = {initial_state: None}  # 记录每个节点的父节点

        for _ in range(max_iterations):
            # 从树中选择一个节点，并在该节点周围扩展
            # random_node = self.select_nearby_node(tree)
            random_node = State(self.tree[-1].time + 1,  # 随机生成time并增加
                                Location(random.uniform(0, 10), random.uniform(0, 10)))
            # 找到树中离random_node最近的节点
            nearest_node = self.nearest(self.tree, random_node)

            # 从nearest_node扩展一步，得到一个新的节点
            new_node = self.steer(nearest_node, random_node)

            # 如果新节点有效（没有碰撞等）
            if new_node is not None:
                # print(new_node)
                if self.state_valid(new_node):
                    new_node.time = nearest_node.time + 1
                    self.tree.append(new_node)
                    parent_map[new_node] = nearest_node  # 记录父节点

                # 如果新节点接近目标，则返回路径
                if self.is_around_goal(new_node, agent_name, 0.1):
                    print("Found path")
                    # plot_rrt_path(self.tree, self.reconstruct_path(self.tree, new_node, parent_map), self.obstacles,
                    #               initial_state.location, goal_state.location)
                    return self.reconstruct_path(self.tree, new_node, parent_map)

        return False  # 如果超过最大迭代次数还未找到路径，则返回失败

    def state_valid(self, state):
        """
        判断状态是否有效，即是否碰到障碍物（考虑障碍物为1x1的方块）
        :param state: 当前状态（包括位置）
        :return: 如果有效返回True，否则返回False
        """
        # 确保位置在合法范围内（地图边界内）
        if state.location.x < 0 or state.location.x >= self.dimension[0]\
                or state.location.y < 0 or state.location.y >= self.dimension[1] - 0.5:
            return False

        # 判断当前点是否在任何障碍物方块内
        for obstacle in self.obstacles:
            # 如果路径点的坐标在障碍物的范围内
            if obstacle[0] - 0.2 < state.location.x < obstacle[0] + 1.2 and \
                    obstacle[1] - 0.7 < state.location.y < obstacle[1] + 0.7:
                return False

        # 检查是否与任何约束冲突（例如，时间或其他状态约束）
        if VertexConstraint(state.time, state.location) in self.constraints.vertex_constraints:
            return False

        return True

    def on_segment(self, p, q, r):
        """检查点 q 是否在线段 pr 上"""
        if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
            return True
        return False

    def orientation(self, p, q, r):
        """计算方向（叉积）"""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counterclockwise

    def do_intersect(self, p1, q1, p2, q2):
        """检查线段 p1q1 与 p2q2 是否相交"""
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        # 普通情况
        if o1 != o2 and o3 != o4:
            return True

        # 特殊情况：共线并且在线段上
        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True
        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True
        if o4 == 0 and self.on_segment(p2, q1, q2):
            return True

        return False

    def line_intersects_obstacle(self, line_start, line_end, obstacles):
        """
        检查从line_start到line_end的路径是否与任何障碍物相交
        :param line_start: 起始点坐标 (x, y)
        :param line_end: 终点坐标 (x, y)
        :param obstacles: 障碍物列表，每个障碍物为 (x, y) 形式的坐标
        :return: 如果路径与任何障碍物相交，返回True，否则返回False
        """
        for obstacle in obstacles:
            obs_x, obs_y = obstacle

            # 生成障碍物的四条边
            obstacle_edges = [
                ((obs_x, obs_y + 0.5), (obs_x + 1, obs_y + 0.5)),  # 上边
                ((obs_x, obs_y - 0.5), (obs_x, obs_y + 0.5)),  # 左边
                ((obs_x + 1, obs_y - 0.5), (obs_x + 1, obs_y + 0.5)),  # 右边
                ((obs_x, obs_y - 0.5), (obs_x + 1, obs_y - 0.5))  # 下边
            ]

            # 检查路径是否与障碍物的四条边相交
            for edge in obstacle_edges:
                if self.do_intersect(line_start, line_end, edge[0], edge[1]):
                    return True  # 如果路径与障碍物的某条边相交，返回True

        return False

    def reconstruct_path(self, tree, current, parent_map):
        """从目标节点反向回溯路径，并倒着计算 time"""
        path = [current]
        # 假设目标节点的 time 为最大值，可以设置为从最大 time 开始递减
        time_counter = len(tree) - 1  # 或者从一个你希望的初始值开始

        # 避免死循环，直接通过 parent_map 回溯父节点
        while parent_map.get(current):  # 如果当前节点有父节点
            current = parent_map[current]
            # 更新节点的时间为递减值
            # current.time = time_counter
            path.append(current)
            # time_counter -= 1  # time 按照顺序递减

        # 将路径反转并返回
        return path[::-1]


