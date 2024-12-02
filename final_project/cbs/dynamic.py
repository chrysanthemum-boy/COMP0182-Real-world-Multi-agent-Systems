import heapq

class DynamicWeightedAStar():
    def __init__(self, env, initial_weight=2.0, weight_decay=0.95):
        """
        Dynamic Weighted A* implementation for CBS low-level search.

        Args:
            env: 环境对象，包含状态有效性和邻居获取等方法。
            initial_weight: 初始启发式权重。
            weight_decay: 每次扩展节点后，权重的衰减因子。
        """
        self.env = env
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors
        self.initial_weight = initial_weight
        self.weight_decay = weight_decay

    def reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from the start to the goal.
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name):
        """
        Perform low-level search for an individual agent using Dynamic Weighted A*.
        """
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1

        open_list = []
        heapq.heappush(open_list, (0, initial_state))  # (priority, state)

        came_from = {}
        g_score = {initial_state: 0}  # Cost from start to current state
        f_score = {initial_state: self.initial_weight * self.admissible_heuristic(initial_state, agent_name)}

        weight = self.initial_weight

        while open_list:
            # Retrieve state with lowest f_score
            _, current = heapq.heappop(open_list)

            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)

            neighbors = self.get_neighbors(current)

            for neighbor in neighbors:
                tentative_g_score = g_score[current] + step_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + weight * self.admissible_heuristic(neighbor, agent_name)

                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

            # Dynamically decay the weight after each node expansion
            weight = max(1.0, weight * self.weight_decay)

        return False  # No valid path found
