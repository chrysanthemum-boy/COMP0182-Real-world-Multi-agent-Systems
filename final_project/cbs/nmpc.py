import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

class NMPC:
    def __init__(self, N, dt, x_goal):
        self.N = N  # Prediction horizon length
        self.dt = dt  # Discrete time step size
        self.x_goal = x_goal  # Target state [x, y, theta]

        # Define state and control input variables
        self.x = ca.MX.sym("x")  # Position x
        self.y = ca.MX.sym("y")  # Position y
        self.theta = ca.MX.sym("theta")  # Orientation angle
        self.v = ca.MX.sym("v")  # Linear velocity
        self.omega = ca.MX.sym("omega")  # Angular velocity

        self.state = ca.vertcat(self.x, self.y, self.theta)
        self.control = ca.vertcat(self.v, self.omega)

        # Dynamics model
        self.f = ca.Function("f", [self.state, self.control],
                             [ca.vertcat(
                                 self.x + self.v * ca.cos(self.theta) * self.dt,
                                 self.y + self.v * ca.sin(self.theta) * self.dt,
                                 self.theta + self.omega * self.dt
                             )])

    def optimize(self, x0):
        """
        Optimize the path using NMPC.
        :param x0: Initial state [x, y, theta]
        :return: Optimized control sequence and state trajectory
        """
        # Define optimization variables
        U = ca.MX.sym("U", 2, self.N)  # Control input [v, omega]
        X = ca.MX.sym("X", 3, self.N + 1)  # State [x, y, theta]

        # Initialize the objective function and constraints
        obj = 0  # Objective function
        g = []  # Constraints
        g.append(X[:, 0] - x0)  # Initial state constraint

        # Construct the objective function and state constraints
        for k in range(self.N):
            x_next = self.f(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next)  # Dynamics constraints

            # Objective function (state error and control input effort)
            obj += ca.mtimes((X[:, k] - self.x_goal).T, (X[:, k] - self.x_goal))  # State error
            obj += ca.mtimes(U[:, k].T, U[:, k])  # Control input effort

        # Define the optimization problem
        opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {
            'x': opt_variables,
            'f': obj,
            'g': ca.vertcat(*g)
        }

        # Set up the solver
        opts = {"ipopt.print_level": 0, "print_time": 0}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Set lower and upper bounds
        lbg = np.zeros((3 * (self.N + 1),))  # Lower bound for dynamics constraints
        ubg = np.zeros((3 * (self.N + 1),))
        lbx = -np.inf * np.ones(opt_variables.shape)  # Lower bound for decision variables
        ubx = np.inf * np.ones(opt_variables.shape)  # Upper bound for decision variables

        # Initial guess
        x0_guess = np.zeros((3 * (self.N + 1),))
        u0_guess = np.zeros((2 * self.N,))
        init_guess = np.concatenate((x0_guess, u0_guess))

        # Solve the optimization problem
        sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        # Extract the solution
        x_opt = np.reshape(sol['x'][:3 * (self.N + 1)], (3, self.N + 1))
        u_opt = np.reshape(sol['x'][3 * (self.N + 1):], (2, self.N))
        return x_opt, u_opt


# Example: Use NMPC to optimize the path
if __name__ == "__main__":
    N = 20  # Prediction horizon length
    dt = 0.1  # Time step size
    x_goal = np.array([5, 5, 0])  # Target state [x, y, theta]
    x0 = np.array([0, 0, 0])  # Initial state

    nmpc = NMPC(N, dt, x_goal)
    x_opt, u_opt = nmpc.optimize(x0)

    # Plot the optimized results
    plt.figure(figsize=(10, 6))
    plt.plot(x_opt[0, :], x_opt[1, :], '-o', label="Optimized Path")
    plt.scatter(x_goal[0], x_goal[1], c='r', label="Goal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("NMPC Optimized Path")
    plt.legend()
    plt.grid()
    plt.show()
