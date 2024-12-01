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
        nx = 3  # State variables: x, y, theta
        nu = 2  # Control inputs: v, omega

        # Decision variables
        X = MX.sym('X', nx, self.horizon + 1)
        U = MX.sym('U', nu, self.horizon)

        # Parameters
        P = MX.sym('P', nx + 2)

        # Dynamics model
        f = lambda x, u: vertcat(
            x[0] + u[0] * cos(x[2]) * self.dt,
            x[1] + u[0] * sin(x[2]) * self.dt,
            x[2] + u[1] * self.dt
        )

        # Objective and constraints
        obj = 0
        g = []

        for t in range(self.horizon):
            obj += (X[0, t] - P[3]) ** 2 + (X[1, t] - P[4]) ** 2  # Distance to target
            obj += 0.1 * U[0, t] ** 2 + 0.1 * U[1, t] ** 2  # Minimize control effort
            g.append(X[:, t + 1] - f(X[:, t], U[:, t]))  # Dynamics constraints

        # Initial state constraint
        g.append(X[:, 0] - P[:3])

        # Flatten constraints
        g = vertcat(*g)

        # Define variable bounds
        self.lbx = [-np.inf] * X.size1() * (self.horizon + 1) + [-self.max_speed] * self.horizon + [
            -self.max_angular_speed] * self.horizon
        self.ubx = [np.inf] * X.size1() * (self.horizon + 1) + [self.max_speed] * self.horizon + [
            self.max_angular_speed] * self.horizon

        # Define lower and upper bounds
        self.lbg = [0] * g.size1()
        self.ubg = [0] * g.size1()

        # Define the NLP problem
        self.nlp = {'x': vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))), 'f': obj, 'g': g, 'p': P}
        self.solver = nlpsol('solver', 'ipopt', self.nlp)

    def plan(self, current_state, target):
        p = vertcat(current_state, target)
        x0 = np.zeros(self.nlp['x'].size1())
        sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        x_traj = sol['x'][:3 * (self.horizon + 1)].reshape((3, self.horizon + 1))
        return x_traj, sol['x'][-2 * self.horizon:]  # 返回路径和控制信号
