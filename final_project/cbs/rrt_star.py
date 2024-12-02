import random
import math
from math import fabs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


class RRTStar:
    def __init__(self, env, search_radius=1, max_iterations=50000, step_size=0.1, goal_bias=0.1):
        self.dimension = env.dimension
        self.obstacles = env.obstacles
        self.constraints = env.constraints
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.is_around_goal = env.is_around_goal
        self.get_neighbors = env.get_neighbors

        self.search_radius = search_radius  # 搜索半径，用于重连
        self.max_iterations = max_iterations  # 最大迭代次数
        self.step_size = step_size  # 扩展步长
        self.goal_bias = goal_bias  # 目标导向的概率

        self.tree = []  # 存储所有的节点
        self.parent_map = {}  # 节点的父节点映射

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
            new_node.time = from_node.time + self.step_size

            # 检查新节点是否与障碍物相交
            if self.line_intersects_obstacle(
                    (from_node.location.x, from_node.location.y),
                    (new_node.location.x, new_node.location.y),
                    self.obstacles):
                return None
            return new_node

    def sample(self, goal_state):
        """随机采样点，加入目标导向"""
        if random.random() < self.goal_bias:
            # 按照目标偏向概率直接采样目标点
            return goal_state
        else:
            # 否则随机采样
            return State(0, Location(random.uniform(0, self.dimension[0]), random.uniform(0, self.dimension[1])))

    def rewire(self, new_node):
        """重新连接新节点周围的邻居"""
        for neighbor in self.tree:
            if self.distance(neighbor, new_node) < self.search_radius:
                new_cost = new_node.time + self.distance(new_node, neighbor)
                if new_cost < neighbor.time:
                    neighbor.time = new_cost
                    self.parent_map[neighbor] = new_node

    def search(self, agent_name):
        """实现 RRT* 搜索"""
        initial_state = self.agent_dict[agent_name]["start"]
        goal_state = self.agent_dict[agent_name]["goal"]
        initial_state.time = 0  # 起点代价为 0

        self.tree = [initial_state]
        self.parent_map = {initial_state: None}
        random.seed(42)

        for _ in range(self.max_iterations):
            # 使用目标导向采样
            random_node = self.sample(goal_state)

            # 找到最近的节点
            nearest_node = self.nearest(self.tree, random_node)

            # 尝试扩展新节点
            new_node = self.steer(nearest_node, random_node)

            if new_node and self.state_valid(new_node):
                self.tree.append(new_node)
                new_node.time = int(nearest_node.time + 1)
                self.parent_map[new_node] = nearest_node

                # Rewire: 优化新节点周围的邻居连接
                self.rewire(new_node)

                # 检查是否到达目标
                if self.is_around_goal(new_node, agent_name, 0.1):
                    print("Found optimized path!")
                    return self.reconstruct_path(self.tree, new_node, self.parent_map)

        return False  # 如果搜索失败，返回 False

    def state_valid(self, state):
        """
        判断状态是否有效，即是否碰到障碍物
        """
        if state.location.x < 0 or state.location.x >= self.dimension[0] - 0.5 \
                or state.location.y < 0 or state.location.y >= self.dimension[1] - 0.5:
            return False

        for obstacle in self.obstacles:
            if obstacle[0] - 0.9 <= state.location.x <= obstacle[0] + 0.9 and \
                    obstacle[1] - 0.9 <= state.location.y <= obstacle[1] + 0.9:
                return False

        return True

    def reconstruct_path(self, tree, current, parent_map):
        """从目标节点反向回溯路径"""
        path = [current]
        print(path)
        while parent_map.get(current):
            current = parent_map[current]
            path.append(current)
        return path[::-1]

    def line_intersects_obstacle(self, line_start, line_end, obstacles):
        """检查路径是否与任何障碍物相交"""
        for obstacle in obstacles:
            obs_x, obs_y = obstacle
            obstacle_edges = [
                ((obs_x, obs_y + 0.9), (obs_x + 1, obs_y + 0.9)),  # 上边
                ((obs_x, obs_y - 0.9), (obs_x, obs_y + 0.9)),  # 左边
                ((obs_x + 1, obs_y - 0.9), (obs_x + 1, obs_y + 0.9)),  # 右边
                ((obs_x, obs_y - 0.9), (obs_x + 1, obs_y - 0.9))  # 下边
            ]
            for edge in obstacle_edges:
                if self.do_intersect(line_start, line_end, edge[0], edge[1]):
                    return True
        return False
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
