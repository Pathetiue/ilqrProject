import numpy as np
from casadi import *
import control # conda install -c conda-forge control slycot
import matplotlib.pyplot as plt


def cas2arr(M):
    rows, cols = M.shape
    arr = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            arr[r,c] = M[r,c]
    return arr

def cart_pole_model(x, u):
    """
    x: list [theta, w, p, v]
    u: list [F]
    """
    # non linear model
    x1_eval = substitute([x1], [theta, w, p, v, F], x + u)[0]
    x1_eval = cas2arr(x1_eval)
    return x1_eval

def state_space(x, u):
    """
    x: list [theta, w, p, v]
    u: list [F]
    """
    # x_k+1 = Ak x_k + Bk u_k
    At = substitute([A], [theta, w, p, v, F], x + u)
    At = cas2arr(At[0])
    Bt = substitute([B], [theta, w, p, v, F], x + u)
    Bt = cas2arr(Bt[0])
    xt = np.array(x).reshape(4, 1)
    ut = np.array(u).reshape(1, 1)
    xt1 = At @ xt + Bt @ ut
    return xt1

def cost_function(x, u):
    # J = x'Qx + u'Ru
    J = 0.5 * (x.T @ Q @ x + u.T @ R @ u)
    return J

def cost_x(x):
    # Jx = Qx
    J = Q @ x
    return J

def cost_xx(x):
    # Jxx = Q
    return Q

def cost_u(u):
    # Ju = Ru
    J = R @ u
    return J

def cost_uu(u):
    # Juu = R
    return R

def ilqr(x, u, iterations):
    """
    x: 4 * N
    u: 1 * N
    """
    min_du = 1e-3
    min_dJ = 1e-4
    min_dx = 1e-4

    back_track = 1

    prev_cost = np.inf
    smallest_cost = np.inf

    for i in range(iterations):
        # Backward pass
        Vx = np.zeros((4, N))
        Vxx = np.zeros((4, 4, N))
        Vx[:, -1] = cost_x(x[:, -1]).reshape(4) * 5
        # print(Vx[:, -1])
        Vxx[:, :, -1] = cost_xx(x[:, -1])

        du = np.zeros((1, N))

        for k in range(N-2, -1, -1):
            # if x[:, k][0] > np.pi:
            #     x[:, k][0] = x[:, k][0] - 2 * np.pi
            # elif x[:, k][0] < -np.pi:
            #     x[:, k][0] = x[:, k][0] + 2 * np.pi

            # Compute Q function
            fu = substitute([B], [theta, w, p, v, F], x[:, k].flatten().tolist() + u[:, k].tolist())[0]
            fu = cas2arr(fu)
            fx = substitute([A], [theta, w, p, v, F], x[:, k].flatten().tolist() + u[:, k].tolist())[0]
            fx = cas2arr(fx)
            
            Qx = cost_x(x[:, k]) + fx.T @ Vx[:, k+1]
            Qu = cost_u(u[:, k]) + fu.T @ Vx[:, k+1]
            Qxx = cost_xx(x[:, k]) + fx.T @ Vxx[:, :, k+1] @ fx
            Quu = cost_uu(u[:, k]) + fu.T @ Vxx[:, :, k+1] @ fu
            Qux = fu.T @ Vxx[:, :, k+1] @ fx

            kt = -np.linalg.inv(Quu) @ Qu
            Kt = -np.linalg.inv(Quu) @ Qux

            du[:, k] = back_track * kt + Kt @ x[:, k]
            Vx[:, k] = Qx - Kt.T @ Quu @ kt
            Vxx[:, :, k] = Qxx - Kt.T @ Quu @ Kt

            # du[:, k] = back_track * kt
            # Vx[:, k] = Qx + Quu @ du[:, k]
            # Vxx[:, :, k] = Qxx

        if np.max(np.abs(du)) < min_du:
            print("Converged at iteration {}, with max du: {}".format(i, np.max(np.abs(du))))
            break

        x_new = np.zeros((4, N))
        u_new = np.zeros((1, N))
        x_new[:, 0] = np.array([np.pi, 0, 0, 0]).reshape(4)

        # Forward pass
        for k in range(N-1):
            u_new[:, k] = u[:, k] + du[:, k]
            x_new[:, k+1] = cart_pole_model(x[:, k].flatten().tolist(), [u[:, k]]).reshape(4)
            # x_new[:, k+1] = state_space(x[:, k].flatten().tolist(), [u[:, k]]).reshape(4)
            pass

        new_cost = np.sum([cost_function(x_new[:, k], u_new[:, k]) for k in range(N)])
        if new_cost < smallest_cost:
            smallest_cost = new_cost
            opt_x = x_new
            opt_u = u_new

        x = x_new
        u = u_new
        
        print("iteration : {}, cost : {}, change in cost: {}".format(i, new_cost, new_cost - prev_cost))
        prev_cost = new_cost
        back_track = max(back_track - back_track / iterations, 1e-3)

    return opt_x, opt_u
    return x, u

if __name__ == "__main__":
    g = 9.8   # Gravity
    h = 0.02  # Sampling time
    mc = 1    # Cart mass
    mp = 0.1  # Pole mass
    l = 0.5   # Half pole length

    theta = SX.sym("theta")
    w = SX.sym("w")
    p = SX.sym("p")
    v = SX.sym("v")

    F = SX.sym("F")

    alpha = (g * sin(theta) + (cos(theta) * ( -F - mp * l * w**2 * sin(theta))) / (mc + mp)) / (l* (4/3 - mp * cos(theta)**2/(mc + mp)))
    acc = (F + mp * l * (w**2 * sin(theta) - alpha*cos(theta))) / (mc + mp)

    theta1 = theta + h * w
    w1 = w + h * alpha
    p1 = p + h * v
    v1 = v + h * acc

    J_theta1_theta = jacobian(theta1, theta)
    J_theta1_w = jacobian(theta1, w)

    x = SX.sym("x", 4,1)
    x[0,0] = theta
    x[1,0] = w
    x[2,0] = p
    x[3,0] = v

    x1 = SX.sym("x1", 4,1)
    x1[0,0] = theta1
    x1[1,0] = w1
    x1[2,0] = p1
    x1[3,0] = v1

    A = jacobian(x1, x)
    B = jacobian(x1, F)

    Q = np.diag([1, 0.1, 1e-9, 1e-9])
    R = np.diag([0.01])

    ref = np.zeros((4,1))

    state = np.array([np.pi, 0, 0, 0]).reshape(4,1)
    state_list = state.flatten().tolist()

    sim_time = 10
    N = int(sim_time / h) + 1
    tspan = [i * h for i in range(N)]

    # Make initial trajectory
    x_traj = np.zeros((4, N))
    x_traj[:,0] = state.reshape(4)
    u = np.zeros((1, N))

    for i in range(1, N):
        x_traj[:, i] = cart_pole_model(x_traj[:,i-1].flatten().tolist(), [u[0, i-1]]).reshape(4)

    x_traj, u = ilqr(x_traj, u, 30)

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(tspan, x_traj[0,:], label="theta")
    ax[1].plot(tspan, u[0,:], label="F")
    ax[0].legend()
    ax[1].legend()
    plt.show()
