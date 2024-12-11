import random
import math
from math import fabs
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


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
    def __init__(self, time, location, cost):
        self.time = time
        self.location = location
        self.cost = cost

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time) + str(self.location.x) + str(self.location.y))

    def is_equal_except_time(self, state):
        return self.location == state.location

    def is_around(self, state, bias):
        return fabs(self.location.x - state.location.x) < bias and fabs(self.location.y - state.location.y) < bias

    def __str__(self):
        return str((self.time, self.location.x, self.location.y, self.cost))


class RRTStar:
    def __init__(self, env, search_radius=1.5, max_iterations=50000, step_size=0.5, goal_bias=0.1):
        self.dimension = env.dimension
        self.obstacles = env.obstacles
        self.constraints = env.constraints
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.is_around_goal = env.is_around_goal
        self.get_neighbors = env.get_neighbors

        self.search_radius = search_radius  # Search radius for rewiring
        self.max_iterations = max_iterations  # Maximum number of iterations
        self.step_size = step_size  # Step size for node expansion
        self.goal_bias = goal_bias  # Probability of biasing towards the goal

        self.tree = []  # Store all nodes
        self.parent_map = {}  # Map of parent nodes

    def distance(self, p1, p2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((p1.location.x - p2.location.x) ** 2 + (p1.location.y - p2.location.y) ** 2)

    def nearest(self, tree, node):
        """Find the closest node in the tree to the given node."""
        return min(tree, key=lambda n: self.distance(n, node))

    def steer(self, from_node, to_node):
        """Move from the current node towards the target node with a given step size."""
        dist = self.distance(from_node, to_node)
        if dist < self.step_size:
            return to_node
        else:
            # Calculate unit vector and use step size to generate a new node
            direction = ((to_node.location.x - from_node.location.x) / dist,
                         (to_node.location.y - from_node.location.y) / dist)
            new_node = State(from_node.time + 1,
                             Location(from_node.location.x + direction[0] * self.step_size,
                                      from_node.location.y + direction[1] * self.step_size),
                             from_node.cost + self.step_size  # Accumulate cost
                             )

            # Check if the new node intersects with any obstacles
            if self.line_intersects_obstacle(
                    (from_node.location.x, from_node.location.y),
                    (new_node.location.x, new_node.location.y),
                    self.obstacles):
                return None
            return new_node

    def sample(self, goal_state):
        """Sample random points, with goal bias."""
        if random.random() < self.goal_bias:
            # Sample the goal point with a certain probability
            return goal_state
        else:
            # Otherwise, sample randomly
            return State(self.tree[-1].time + 1,
                         Location(random.uniform(0, self.dimension[0]), random.uniform(0, self.dimension[1])),
                         cost=self.tree[-1].cost)

    def rewire(self, new_node):
        """Rewire neighbors around the new node."""
        for neighbor in self.tree:
            if self.distance(neighbor, new_node) < self.search_radius:
                new_cost = new_node.cost + self.distance(new_node, neighbor)
                if new_cost < neighbor.cost:
                    neighbor.cost = new_cost
                    self.parent_map[neighbor] = new_node

    def search(self, agent_name):
        """Perform RRT* search."""
        initial_state = self.agent_dict[agent_name]["start"]
        goal_state = self.agent_dict[agent_name]["goal"]
        initial_state.time = 0  # Start time is 0
        initial_state.cost = 0  # Start cost is 0
        self.tree = [initial_state]
        self.parent_map = {initial_state: None}
        random.seed(42)

        for _ in range(self.max_iterations):
            # Perform goal-biased sampling
            random_node = self.sample(goal_state)

            # Find the nearest node in the tree
            nearest_node = self.nearest(self.tree, random_node)

            # Attempt to expand to a new node
            new_node = self.steer(nearest_node, random_node)

            if new_node and self.state_valid(new_node):
                new_node.time = nearest_node.time + 1
                self.tree.append(new_node)
                self.parent_map[new_node] = nearest_node

                # Rewire: Optimize connections around the new node
                # self.rewire(new_node)

                # Check if the goal is reached
                if self.is_around_goal(new_node, agent_name, 0.1):
                    print("Found optimized path!")
                    return self.reconstruct_path(self.tree, new_node, self.parent_map)

        return False  # Return False if search fails

    def state_valid(self, state):
        """Check if the state is valid, i.e., does not collide with obstacles."""
        if state.location.x < 0 or state.location.x >= self.dimension[0] - 0.9 \
                or state.location.y < 0 or state.location.y >= self.dimension[1] - 0.9:
            return False

        for obstacle in self.obstacles:
            if obstacle[0] - 0.9 <= state.location.x <= obstacle[0] + 0.9 and \
                    obstacle[1] - 0.9 <= state.location.y <= obstacle[1] + 0.9:
                return False

        return True

    def reconstruct_path(self, tree, current, parent_map):
        """Backtrace the path from the goal node."""
        path = [current]
        time_counter = len(tree) - 1  # Start from the last node in the tree

        # Avoid infinite loops by tracing back via parent_map
        while parent_map.get(current):  # If the current node has a parent
            current = parent_map[current]
            path.append(current)

        # Reverse the path and return
        return path[::-1]

    def line_intersects_obstacle(self, line_start, line_end, obstacles):
        """Check if the path intersects with any obstacle."""
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
        """Check if point q is on segment pr."""
        if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
            return True
        return False

    def orientation(self, p, q, r):
        """Calculate the orientation (cross product)."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        elif val > 0:
            return 1  # Clockwise
        else:
            return 2  # Counterclockwise

    def do_intersect(self, p1, q1, p2, q2):
        """Check if line segment p1q1 intersects with p2q2."""
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases: Collinear and on segment
        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True
        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True
        if o4 == 0 and self.on_segment(p2, q1, q2):
            return True

        return False
