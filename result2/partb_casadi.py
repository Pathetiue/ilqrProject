import os
import numpy as np
import casadi as cs
from casadi import SX
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class Config:
    def __init__(self):
        self.g = 9.8
        self.m_cart = 1.0
        self.m_pole = 0.1
        self.l = 1.0
        self.J = (self.m_pole * self.l**2)
        self.h = 0.02
        self.T = 5
        self.mpc_horizon = 200
        self.u_max = 15
        self.OnlyProcessData = True
        self.RunControl = False

class Visualizer:
    def __init__(self, states, conf):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-6, 1)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('p')
        self.ax.grid()
        self.cart, = self.ax.plot([], [], 'o-', lw=2,color='k')
        self.pole, = self.ax.plot([], [], '-', lw=2,color='k')
        self.states = states
        self.conf = conf

        self.save_dir = "./"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def init(self):
        self.cart.set_data([], [])
        self.pole.set_data([], [])
        return self.cart, self.pole

    def update(self, frame):
        p = self.states[frame, 2]
        theta = self.states[frame, 0]
        self.cart.set_data([p - 0.45, p + 0.45], [0, 0])
        self.pole.set_data([p, p + self.conf.l * np.sin(np.pi - theta)], [0, -self.conf.l * np.cos(np.pi - theta)])
        return self.cart, self.pole
    def generate_animation(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.states), interval=self.conf.h * 1000)
        animation.ImageMagickWriter(fps=20)
        ani.save(self.save_dir + 'animation.gif', writer='pillow')
        plt.show()

class CartPole:
    def __init__(self, conf):
        self.n_x = 4
        self.n_u = 1
        self.conf = conf
        self.index_x = 2
        self.index_x_dot = 3
        self.index_theta = 0
        self.index_theta_dot = 1
        self.guess = 0

    def step(self, x, u):
        x_dot = x[:, self.index_x_dot]
        theta = x[:, self.index_theta]
        theta_dot = x[:, self.index_theta_dot]
        control = u[:, 0]
        theta_acc = (self.conf.g * np.sin(theta) + np.cos(theta) * (
                    -control - self.conf.m_pole * self.conf.l * theta_dot ** 2 * np.sin(theta)) / (
                             self.conf.m_cart + self.conf.m_pole)) / (
                            self.conf.l * (4 / 3 - self.conf.m_pole * np.cos(theta) ** 2 / (
                                self.conf.m_cart + self.conf.m_pole)))
        x_acc = (control + self.conf.m_pole * self.conf.l * (
                    theta_dot ** 2 * np.sin(theta) - theta_acc * np.cos(theta))) / (
                            self.conf.m_cart + self.conf.m_pole)
        diff = 0 * x
        diff[:, self.index_x] = x_dot
        diff[:, self.index_x_dot] = x_acc
        diff[:, self.index_theta] = theta_dot
        diff[:, self.index_theta_dot] = theta_acc
        next_x = x + self.conf.h * diff
        return next_x

    def mpc(self, x0):
        states = SX.sym('state', self.conf.mpc_horizon, self.n_x)
        controls = SX.sym('input', self.conf.mpc_horizon - 1, self.n_u)

        var_list = [states, controls]
        var_name = ['states', 'inputs']
        vars = cs.vertcat(*[cs.reshape(e, -1, 1) for e in var_list])
        pack_vars = cs.Function('pack_vars', var_list, [vars],
                                                 var_name, ['flat'])
        unpack_vars = cs.Function('unpack_vars', [vars], var_list,
                                                   ['flat'], var_name)
        # constraints
        lbs = unpack_vars(flat=-float('inf'))
        ups = unpack_vars(flat=float('inf'))
        lbs['inputs'][:, 0] = -self.conf.u_max
        ups['inputs'][:, 0] = self.conf.u_max
        lbs['states'][0, self.index_x] = x0[0, self.index_x]
        ups['states'][0, self.index_x] = x0[0, self.index_x]
        lbs['states'][0, self.index_x_dot] = x0[0, self.index_x_dot]
        ups['states'][0, self.index_x_dot] = x0[0, self.index_x_dot]
        lbs['states'][0, self.index_theta] = x0[0, self.index_theta]
        ups['states'][0, self.index_theta] = x0[0, self.index_theta]
        lbs['states'][0, self.index_theta_dot] = x0[0, self.index_theta_dot]
        ups['states'][0, self.index_theta_dot] = x0[0, self.index_theta_dot]

        # dynamics
        X0 = states[0:self.conf.mpc_horizon - 1, :]
        X1 = states[1:self.conf.mpc_horizon, :]
        next_x = self.step(X0, controls) - X1
        next_x = cs.reshape(next_x, -1, 1)


        x_dot = states[:, self.index_x_dot]
        theta = states[:, self.index_theta]
        theta_dot = states[:, self.index_theta_dot]
        kinetic_energy = 0.5 * (self.conf.m_cart + self.conf.m_pole) * x_dot ** 2
        kinetic_energy += self.conf.m_pole * x_dot * theta_dot * self.conf.l * cs.cos(theta - np.pi)
        kinetic_energy += 0.5 * (self.conf.m_pole * self.conf.l ** 2) * theta_dot ** 2
        potential_energy = -self.conf.m_pole * self.conf.g * self.conf.l * cs.cos(theta - np.pi)
        cost = cs.sum1(kinetic_energy - potential_energy)
        for i in range(self.conf.mpc_horizon-1):
            cost += controls[i, 0] * controls[i, 0] * 0.05
        
        # cost = 0
        # ref = np.array([0,0,0,0])
        # for i in range(self.conf.mpc_horizon-1):
        #     cost += states[i, self.index_x_dot] * states[i, self.index_x_dot] * 0.5
        #     cost += states[i, self.index_theta] * states[i, self.index_theta] * 5
        #     cost += states[i, self.index_theta_dot] * states[i, self.index_theta_dot] * 0.5
        #     cost += controls[i, 0] * controls[i, 0] * 0.05
        # cost += states[self.conf.mpc_horizon-1, self.index_x_dot] * states[self.conf.mpc_horizon-1, self.index_x_dot] * 0.5
        # cost += states[self.conf.mpc_horizon-1, self.index_theta] * states[self.conf.mpc_horizon-1, self.index_theta] * 5
        # cost += states[self.conf.mpc_horizon-1, self.index_theta_dot] * states[self.conf.mpc_horizon-1, self.index_theta_dot] * 0.5

        opts = {'ipopt.print_level': 0, 'print_time': False}
        solver = cs.nlpsol('solver', 'ipopt', {'x': vars, 'f': cost, 'g': next_x}, opts)

        sol = solver(x0=self.guess, lbg=0.0, ubg=0.0,
                        lbx=pack_vars(**lbs)['flat'],
                        ubx=pack_vars(**ups)['flat'])
        self.guess = sol['x']
        results = unpack_vars(flat=sol['x'])

        return results['inputs'][0]


conf = Config()
cartpole = CartPole(conf)



if conf.RunControl:
    state = np.array([[np.pi, 0, 0, 0]])  # [theta, theta_dot, x, x_dot]
    states_data  = [state]
    controls_data = []
    for t in np.arange(0, conf.T, conf.h):
        start_time = time.time()
        control = cartpole.mpc(state)
        print("solving time: ", time.time() - start_time)
        state = cartpole.step(state.reshape(1, -1), np.array([[control]]))
        states_data.append(state)
        controls_data.append(control)
        print("time: ", t)

    states_data = np.array(states_data).squeeze(1)
    controls_data = np.array(controls_data).reshape(-1, 1)
    np.save("states.npy", states_data)
    np.save("actions.npy", controls_data)

if conf.OnlyProcessData:
    states_data = np.load("states.npy")
    controls_data = np.load("actions.npy")


# vis = Visualizer(states_data, conf)
# vis.generate_animation()

fig, axs = plt.subplots(4)
axs[0].plot(states_data[:, 0], 'k')
axs[0].set_ylabel('theta')
axs[1].plot(states_data[:, 1], 'k')
axs[1].set_ylabel('theta_dot')
axs[2].plot(states_data[:, 2], 'k')
axs[2].set_ylabel('p')
axs[3].plot(states_data[:, 3], 'k')
axs[3].set_ylabel('p_dot')
axs[3].set_xlabel('t')
fig.tight_layout() 
# fig.legend()
plt.savefig("./states.png")
plt.show()

fig = plt.figure()
plt.plot(controls_data, 'k')
x_min = np.min(controls_data[:,0])
x_max = np.max(controls_data[:,0])
plt.axhline(y=-15.0, color='r', linestyle='--')
plt.axhline(y=15.0, color='r', linestyle='--')
plt.xlabel('t')
plt.ylabel('F')
plt.savefig("./actions.png")
plt.show()




