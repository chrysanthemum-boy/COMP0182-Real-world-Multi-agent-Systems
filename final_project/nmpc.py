import numpy as np
from casadi import *
import matplotlib.pyplot as plt


class NMPCPlanner:
    def __init__(self, horizon=10, dt=0.1, max_speed=1.0, max_angular_speed=1.0):
        self.horizon = horizon
        self.dt = dt
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self._build_optimizer()

    def _build_optimizer(self):
        nx = 3  # 状态变量: x, y, theta
        nu = 2  # 控制变量: v, omega

        # 决策变量
        X = MX.sym('X', nx, self.horizon + 1)
        U = MX.sym('U', nu, self.horizon)

        # 参数
        P = MX.sym('P', nx + 2)  # 当前状态和目标点

        # 动力学模型
        f = lambda x, u: vertcat(
            x[0] + u[0] * cos(x[2]) * self.dt,
            x[1] + u[0] * sin(x[2]) * self.dt,
            x[2] + u[1] * self.dt
        )

        obj = 0
        g = []
        for t in range(self.horizon):
            obj += (X[0, t] - P[3])**2 + (X[1, t] - P[4])**2
            obj += 0.1 * U[0, t]**2 + 0.1 * U[1, t]**2
            g.append(X[:, t + 1] - f(X[:, t], U[:, t]))

        g.append(X[:, 0] - P[:3])  # 初始状态约束

        self.nlp = {'x': vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))), 'f': obj, 'g': vertcat(*g), 'p': P}
        self.solver = nlpsol('solver', 'ipopt', self.nlp)

        self.lbx = [-np.inf] * X.size1() * (self.horizon + 1) + [-self.max_speed] * self.horizon + [-self.max_angular_speed] * self.horizon
        self.ubx = [np.inf] * X.size1() * (self.horizon + 1) + [self.max_speed] * self.horizon + [self.max_angular_speed] * self.horizon
        self.lbg = [0] * g.size1()
        self.ubg = [0] * g.size1()

    def plan(self, current_state, target):
        p = vertcat(current_state, target)
        x0 = np.zeros(self.nlp['x'].size1())
        sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        x_traj = sol['x'][:3 * (self.horizon + 1)].reshape((3, self.horizon + 1))
        return x_traj, sol['x'][-2 * self.horizon:]  # 返回路径和控制信号



if __name__ == "__main__":
    planner = NMPCPlanner()

    # Ensure types are compatible with CasADi
    current_state = DM([0, 0, 0])  # x, y, theta
    target = DM([5, 5])  # Target position (x, y)

    path = [current_state[:2].full().flatten().tolist()]
    for step in range(50):
        # Combine state and target
        p = vertcat(current_state, target)

        # Plan
        linear_velocity, angular_velocity = planner.plan(p)
        print(f"Step {step}: Linear Velocity = {linear_velocity}, Angular Velocity = {angular_velocity}")

        # Simulate motion
        current_state[0] += linear_velocity * np.cos(current_state[2]) * planner.dt
        current_state[1] += linear_velocity * np.sin(current_state[2]) * planner.dt
        current_state[2] += angular_velocity * planner.dt
        current_state[2] = fmod(current_state[2] + 2 * np.pi, 2 * np.pi)  # Normalize angle

        # Log updated state
        print(f"Updated State: {current_state}")

        # Append path
        path.append(current_state[:2].full().flatten().tolist())

        # Check distance to target
        distance_to_target = np.linalg.norm(np.array(current_state[:2].full()) - np.array(target.full()))
        print(f"Distance to Target: {distance_to_target}")

        if distance_to_target < 0.1:
            print("Target reached.")
            break

    # Convert target to NumPy array
    target_np = target.full().flatten()

    # Plot path
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], marker='o', label='Planned Path')
    plt.scatter(target_np[0], target_np[1], color='red', label='Target')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('NMPC Path Planning')
    plt.show()

