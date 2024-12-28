import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        x_min = np.min(states[:,2])
        x_max = np.max(states[:,2])
        self.ax.set_xlim(x_min, x_max)
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
        # plt.show()

if __name__ == "__main__":
    states = np.load("states_p2.npy")
    ctrl = np.load("u_res_p2.npy")
    cfg = Config()
    viz = Visualizer(states=states.T, conf=cfg)
    viz.generate_animation()

    sim_time = 30
    n_sim_step = int((sim_time / 0.02) + 1)
    tspan = np.linspace(0, sim_time, n_sim_step)

    x_res = states
    u_res = ctrl

    fig, ax = plt.subplots(3, 2)
    ax[2,1].axis("off")
    fig.suptitle("Prediction Horizon: {}s".format(2))    
    ax[0, 0].plot(tspan, x_res[0,:], label="theta")
    ax[0, 0].set_ylabel("theta (rad)")
    ax[1, 0].plot(tspan, x_res[1,:], label="w")
    ax[1, 0].set_ylabel("w (rad/s)")
    ax[0, 1].plot(tspan, x_res[2,:], label="p")
    ax[0, 1].set_ylabel("p (m)")
    ax[1, 1].plot(tspan, x_res[3,:], label="v")
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_ylabel("v (m/s)")
    ax[2, 0].plot(tspan, u_res[0,:], label="F")
    ax[2, 0].set_xlabel("time (s)")
    ax[2, 0].set_ylabel("F (N)")
    ax[0, 0].legend()
    ax[1, 0].legend()
    ax[0, 1].legend()
    ax[1, 1].legend()
    ax[2, 0].legend()
    
    plt.show()