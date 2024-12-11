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

        self.search_radius = search_radius  # Search radius for rewiring
        self.max_iterations = max_iterations  # Maximum iterations
        self.step_size = step_size  # Step size for expansion
        self.goal_bias = goal_bias  # Probability for goal-directed sampling

        self.tree = []  # Stores all nodes
        self.parent_map = {}  # Maps each node to its parent

    def distance(self, p1, p2):
        """Compute Euclidean distance between two points."""
        return math.sqrt((p1.location.x - p2.location.x) ** 2 + (p1.location.y - p2.location.y) ** 2)

    def nearest(self, tree, node):
        """Find the nearest node in the tree to the target node."""
        return min(tree, key=lambda n: self.distance(n, node))

    def steer(self, from_node, to_node):
        """Move from the current node towards the target node."""
        dist = self.distance(from_node, to_node)
        if dist < self.step_size:
            return to_node
        else:
            # Calculate unit vector and multiply by step size to get new node position
            direction = ((to_node.location.x - from_node.location.x) / dist,
                         (to_node.location.y - from_node.location.y) / dist)
            new_node = State(from_node.time + 1,
                             Location(from_node.location.x + direction[0] * self.step_size,
                                      from_node.location.y + direction[1] * self.step_size))
            new_node.time = from_node.time + self.step_size

            # Check if the new node intersects with any obstacles
            if self.line_intersects_obstacle(
                    (from_node.location.x, from_node.location.y),
                    (new_node.location.x, new_node.location.y),
                    self.obstacles):
                return None
            return new_node

    def sample(self, goal_state):
        """Randomly sample a point, with goal-directed sampling."""
        if random.random() < self.goal_bias:
            # With a certain probability, sample the goal point directly
            return goal_state
        else:
            # Otherwise, sample a random point
            return State(0, Location(random.uniform(0, self.dimension[0]), random.uniform(0, self.dimension[1])))

    def rewire(self, new_node):
        """Reconnect neighbors of the new node for optimization."""
        for neighbor in self.tree:
            if self.distance(neighbor, new_node) < self.search_radius:
                new_cost = new_node.time + 1
                if new_cost < neighbor.time:
                    neighbor.time = new_cost
                    self.parent_map[neighbor] = new_node

    def search(self, agent_name):
        """Implement the RRT* search."""
        initial_state = self.agent_dict[agent_name]["start"]
        goal_state = self.agent_dict[agent_name]["goal"]
        initial_state.time = 0  # Initialize cost for start node

        self.tree = [initial_state]
        self.parent_map = {initial_state: None}
        random.seed(42)

        for _ in range(self.max_iterations):
            # Perform goal-directed sampling
            random_node = self.sample(goal_state)

            # Find the nearest node
            nearest_node = self.nearest(self.tree, random_node)

            # Try to expand a new node
            new_node = self.steer(nearest_node, random_node)

            if new_node and self.state_valid(new_node):
                self.tree.append(new_node)
                new_node.time = int(nearest_node.time + 1)
                self.parent_map[new_node] = nearest_node

                # Rewire: Optimize connections of neighbors
                self.rewire(new_node)

                # Check if the goal is reached
                if self.is_around_goal(new_node, agent_name, 0.1):
                    print("Found optimized path!")
                    return self.reconstruct_path(self.tree, new_node, self.parent_map)

        return False  # Return False if the search fails

    def state_valid(self, state):
        """Check if the state is valid (i.e., not colliding with obstacles)."""
        if state.location.x < 0 or state.location.x >= self.dimension[0] - 0.9 \
                or state.location.y < 0 or state.location.y >= self.dimension[1] - 0.9:
            return False

        for obstacle in self.obstacles:
            if obstacle[0] - 0.9 <= state.location.x <= obstacle[0] + 0.9 and \
                    obstacle[1] - 0.9 <= state.location.y <= obstacle[1] + 0.9:
                return False

        return True

    def reconstruct_path(self, tree, current, parent_map):
        """Trace back the path from the goal to the start."""
        path = [current]
        print(path)
        while parent_map.get(current):
            current = parent_map[current]
            path.append(current)
        return path[::-1]

    def line_intersects_obstacle(self, line_start, line_end, obstacles):
        """Check if the path intersects with any obstacles."""
        for obstacle in obstacles:
            obs_x, obs_y = obstacle
            obstacle_edges = [
                ((obs_x, obs_y + 0.9), (obs_x + 1, obs_y + 0.9)),  # Top edge
                ((obs_x, obs_y - 0.9), (obs_x, obs_y + 0.9)),  # Left edge
                ((obs_x + 1, obs_y - 0.9), (obs_x + 1, obs_y + 0.9)),  # Right edge
                ((obs_x, obs_y - 0.9), (obs_x + 1, obs_y - 0.9))  # Bottom edge
            ]
            for edge in obstacle_edges:
                if self.do_intersect(line_start, line_end, edge[0], edge[1]):
                    return True
        return False

    def on_segment(self, p, q, r):
        """Check if point q lies on segment pr."""
        if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
            return True
        return False

    def orientation(self, p, q, r):
        """Compute the orientation (cross product)."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counterclockwise

    def do_intersect(self, p1, q1, p2, q2):
        """Check if line segments p1q1 and p2q2 intersect."""
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special case: collinear and lying on the segment
        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True
        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True
        if o4 == 0 and self.on_segment(p2, q1, q2):
            return True

        return False
