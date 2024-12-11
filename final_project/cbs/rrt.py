import random
import math
from math import fabs
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_rrt_path(tree, path, obstacles=None, start=None, goal=None):
    """
    Plot the RRT-generated path along with obstacles, start, and goal points.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw obstacles (if any)
    if obstacles:
        for obs in obstacles:
            ax.add_patch(plt.Rectangle((obs[0], obs[1] - 0.5), 1, 0.5, color="gray", alpha=0.5))

    # Plot start and goal points
    if start:
        ax.plot(start.x, start.y, 'go', label='Start')
    if goal:
        ax.plot(goal.x, goal.y, 'ro', label='Goal')

    # Plot the path
    if path:
        path_x = [state.location.x for state in path]
        path_y = [state.location.y for state in path]
        ax.plot(path_x, path_y, 'b-', label='Path')

    # Plot the tree
    tree_x = [state.location.x for state in tree]
    tree_y = [state.location.y for state in tree]
    ax.plot(tree_x, tree_y, 'k.', label='RRT Tree')

    # Set display parameters
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("RRT Path Planning")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # Show the plot
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
        self.admissible_heuristic = env.admissible_heuristic  # Optional, can be used if needed
        self.is_at_goal = env.is_at_goal
        self.is_around_goal = env.is_around_goal
        self.get_neighbors = env.get_neighbors  # Not necessarily required in RRT
        self.max_iterations = 50000  # Maximum expansion iterations
        self.step_size = 0.5  # Step size
        self.tree = []

    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1.location.x - p2.location.x) ** 2 + (p1.location.y - p2.location.y) ** 2)

    def nearest(self, tree, node):
        """Find the closest node in the tree to the target node."""
        return min(tree, key=lambda n: self.distance(n, node))

    def steer(self, from_node, to_node):
        """Move from the current node toward the target node with a step size."""
        dist = self.distance(from_node, to_node)
        if dist < self.step_size:
            return to_node
        else:
            # Calculate unit vector and move step size to generate a new node
            direction = ((to_node.location.x - from_node.location.x) / dist,
                         (to_node.location.y - from_node.location.y) / dist)
            new_node = State(from_node.time + 1,
                             Location(from_node.location.x + direction[0] * self.step_size,
                                      from_node.location.y + direction[1] * self.step_size))

            # Check if the path intersects with any obstacles
            if self.line_intersects_obstacle(
                    (from_node.location.x, from_node.location.y),
                    (new_node.location.x, new_node.location.y),
                    self.obstacles):
                return None  # If the path intersects an obstacle, return None

            return new_node

    def search(self, agent_name):
        """
        Low-level RRT search
        """
        initial_state = self.agent_dict[agent_name]["start"]
        goal_state = self.agent_dict[agent_name]["goal"]
        max_iterations = self.max_iterations

        self.tree = [initial_state]  # Initialize the tree with the root node
        parent_map = {initial_state: None}  # Map to store parent of each node
        random.seed(42)
        for _ in range(max_iterations):
            # Select a nearby node and expand around it
            random_node = State(self.tree[-1].time + 1,  # Generate a random time increment
                                Location(random.randint(0, 10), random.randint(0, 10)))
            # Find the nearest node in the tree to the random node
            nearest_node = self.nearest(self.tree, random_node)

            # Expand one step from the nearest node
            new_node = self.steer(nearest_node, random_node)

            # If the new node is valid (no collisions, etc.)
            if new_node is not None:
                if self.state_valid(new_node):
                    new_node.time = nearest_node.time + 1
                    self.tree.append(new_node)
                    parent_map[new_node] = nearest_node  # Record the parent node

                # If the new node is close to the goal, return the path
                if self.is_around_goal(new_node, agent_name, 0.1):
                    print("Found path")
                    return self.reconstruct_path(self.tree, new_node, parent_map)

        return False  # Return False if no path is found within the maximum iterations

    def state_valid(self, state):
        """
        Check if the state is valid, i.e., it does not collide with any obstacles.
        :param state: Current state (including position)
        :return: True if valid, False otherwise
        """
        # Ensure the position is within valid range (map boundaries)
        if state.location.x < 0 or state.location.x >= self.dimension[0] \
                or state.location.y < 0 or state.location.y >= self.dimension[1] - 0.5:
            return False

        # Check if the position is inside any obstacle
        for obstacle in self.obstacles:
            if obstacle[0] - 0.9 < state.location.x < obstacle[0] + 0.9 and \
                    obstacle[1] - 0.9 < state.location.y < obstacle[1] + 0.9:
                return False

        # Check if the state violates any constraints
        if VertexConstraint(state.time, state.location) in self.constraints.vertex_constraints:
            return False

        return True

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
        """Check if line segment p1q1 intersects p2q2."""
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

    def line_intersects_obstacle(self, line_start, line_end, obstacles):
        """
        Check if the path from line_start to line_end intersects with any obstacle.
        :param line_start: Start point coordinates (x, y)
        :param line_end: End point coordinates (x, y)
        :param obstacles: List of obstacles, each represented as (x, y)
        :return: True if the path intersects any obstacle, False otherwise
        """
        for obstacle in obstacles:
            obs_x, obs_y = obstacle

            # Define the four edges of the obstacle
            obstacle_edges = [
                ((obs_x, obs_y + 0.9), (obs_x + 1, obs_y + 0.9)),  # Top edge
                ((obs_x, obs_y - 0.9), (obs_x, obs_y + 0.9)),  # Left edge
                ((obs_x + 1, obs_y - 0.9), (obs_x + 1, obs_y + 0.9)),  # Right edge
                ((obs_x, obs_y - 0.9), (obs_x + 1, obs_y - 0.9))  # Bottom edge
            ]

            # Check if the path intersects with any edge of the obstacle
            for edge in obstacle_edges:
                if self.do_intersect(line_start, line_end, edge[0], edge[1]):
                    return True  # Return True if the path intersects any edge

        return False

    def reconstruct_path(self, tree, current, parent_map):
        """Backtrace the path from the goal node, adjusting time in reverse order."""
        path = [current]
        time_counter = len(tree) - 1  # Or start from an initial value

        # Avoid infinite loops, trace back using parent_map
        while parent_map.get(current):  # If the current node has a parent
            current = parent_map[current]
            path.append(current)

        # Reverse the path and return it
        return path[::-1]
