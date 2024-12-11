import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

class NMPC:
    def __init__(self, N, dt, x_goal):
        self.N = N  # 预测时域长度
        self.dt = dt  # 离散时间步长
        self.x_goal = x_goal  # 目标状态 [x, y, theta]

        # 定义状态和控制输入变量
        self.x = ca.MX.sym("x")  # 位置 x
        self.y = ca.MX.sym("y")  # 位置 y
        self.theta = ca.MX.sym("theta")  # 朝向角
        self.v = ca.MX.sym("v")  # 线速度
        self.omega = ca.MX.sym("omega")  # 角速度

        self.state = ca.vertcat(self.x, self.y, self.theta)
        self.control = ca.vertcat(self.v, self.omega)

        # 动力学模型
        self.f = ca.Function("f", [self.state, self.control],
                             [ca.vertcat(
                                 self.x + self.v * ca.cos(self.theta) * self.dt,
                                 self.y + self.v * ca.sin(self.theta) * self.dt,
                                 self.theta + self.omega * self.dt
                             )])

    def optimize(self, x0):
        """
        使用 NMPC 优化路径
        :param x0: 初始状态 [x, y, theta]
        :return: 优化后的控制序列和状态轨迹
        """
        # 定义优化变量
        U = ca.MX.sym("U", 2, self.N)  # 控制输入 [v, omega]
        X = ca.MX.sym("X", 3, self.N + 1)  # 状态 [x, y, theta]

        # 初始化目标函数和约束
        obj = 0  # 目标函数
        g = []  # 约束
        g.append(X[:, 0] - x0)  # 初始状态约束

        # 构造目标函数和状态约束
        for k in range(self.N):
            x_next = self.f(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next)  # 动力学约束

            # 目标函数（状态误差和控制输入能量）
            obj += ca.mtimes((X[:, k] - self.x_goal).T, (X[:, k] - self.x_goal))  # 状态误差
            obj += ca.mtimes(U[:, k].T, U[:, k])  # 控制输入能量

        # 定义优化问题
        opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {
            'x': opt_variables,
            'f': obj,
            'g': ca.vertcat(*g)
        }

        # 设置求解器
        opts = {"ipopt.print_level": 0, "print_time": 0}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # 设置上下界
        lbg = np.zeros((3 * (self.N + 1),))  # 动力学约束的下界
        ubg = np.zeros((3 * (self.N + 1),))
        lbx = -np.inf * np.ones(opt_variables.shape)  # 决策变量下界
        ubx = np.inf * np.ones(opt_variables.shape)  # 决策变量上界

        # 初始值
        x0_guess = np.zeros((3 * (self.N + 1),))
        u0_guess = np.zeros((2 * self.N,))
        init_guess = np.concatenate((x0_guess, u0_guess))

        # 求解优化问题
        sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        # 提取解
        x_opt = np.reshape(sol['x'][:3 * (self.N + 1)], (3, self.N + 1))
        u_opt = np.reshape(sol['x'][3 * (self.N + 1):], (2, self.N))
        return x_opt, u_opt


# 示例：使用 NMPC 优化路径
if __name__ == "__main__":
    N = 20  # 预测时域长度
    dt = 0.1  # 时间步长
    x_goal = np.array([5, 5, 0])  # 目标状态 [x, y, theta]
    x0 = np.array([0, 0, 0])  # 初始状态

    nmpc = NMPC(N, dt, x_goal)
    x_opt, u_opt = nmpc.optimize(x0)

    # 绘制优化结果
    plt.figure(figsize=(10, 6))
    plt.plot(x_opt[0, :], x_opt[1, :], '-o', label="Optimized Path")
    plt.scatter(x_goal[0], x_goal[1], c='r', label="Goal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("NMPC Optimized Path")
    plt.legend()
    plt.grid()
    plt.show()
