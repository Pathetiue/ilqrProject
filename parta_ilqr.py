import numpy as np
np.seterr(all="raise")
np.set_printoptions(precision=3)
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
    J =  R @ u
    return J

def cost_uu(u):
    # Juu = R
    return R

def wrap_angle(angle):
    if angle > np.pi:
        angle = angle % (2 * np.pi)
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle = angle % (-2 * np.pi)
        angle += 2 * np.pi
    return angle

def ilqr(x, u, iterations):
    """
    x: 4 * N
    u: 1 * N
    """
    min_du = 1e-3
    min_dJ = 1e-4
    min_dx = 1e-4
    back_track_base = 1
    back_track = back_track_base

    prev_cost = np.inf
    smallest_cost = np.inf

    itr = 0
    back_itr = 0
    tried_reverse_search = False
    while itr < iterations:
        # Backward pass
        Vx = np.zeros((4, N))
        Vxx = np.zeros((4, 4, N))
        Vx[:, -1] = Qf @ x[:, -1]
        Vxx[:, :, -1] = Qf

        du = np.zeros((1, N))
        kts = [None] * N
        Kts = [None] * N

        for k in range(N-2, -1, -1):
            x[:, k][0] = wrap_angle(x[:, k][0])

            # Compute Q function
            fx = substitute([A], [theta, w, p, v, F], x[:, k].flatten().tolist() + u[:, k].tolist())[0]
            fx = cas2arr(fx)
            fu = substitute([B], [theta, w, p, v, F], x[:, k].flatten().tolist() + u[:, k].tolist())[0]
            fu = cas2arr(fu)

            Qx = cost_x(x[:, k]) + fx.T @ Vx[:, k+1]
            Qu = cost_u(u[:, k]) + fu.T @ Vx[:, k+1]
            Qx = cost_x(x[:, k]) + (fx.T @ Vx[:, k+1]).T
            Qu = cost_u(u[:, k]) + (fu.T @ Vx[:, k+1]).T
            Qxx = cost_xx(x[:, k]) + fx.T @ Vxx[:, :, k+1] @ fx
            Quu = cost_uu(u[:, k]) + fu.T @ Vxx[:, :, k+1] @ fu + np.eye(fu.shape[1]) * 0.1
            Qux = fu.T @ Vxx[:, :, k+1] @ fx

            kt = -np.linalg.inv(Quu) @ Qu
            Kt = -np.linalg.inv(Quu) @ Qux

            kts[k] = kt
            Kts[k] = Kt

            du[:, k] = kt + Kt @ x[:, k]
            Vx[:, k] = Qx - Kt.T @ Quu @ kt
            Vxx[:, :, k] = Qxx - Kt.T @ Quu @ Kt

        x_new = np.zeros((4, N))
        u_new = np.zeros((1, N))
        x_new[:, 0] = x[:, 0].reshape(4)

        # Forward pass
        for k in range(N-1):
            du = back_track * kts[k] + Kts[k] @ x_new[:, k]
            u_new[:, k] = u[:, k] + du
            u_new[:, k] = np.clip(u_new[:, k], -50, 50)

            x_new[:, k+1] = cart_pole_model(x_new[:, k].flatten().tolist(), u_new[:, k].tolist()).reshape(4)
            x_new[:, k+1][0] = wrap_angle(x_new[:, k+1][0])
            pass

        new_cost = np.sum([cost_function(x_new[:, k], u_new[:, k]) for k in range(N-1)])
        last_cost = x_new[:, -1].T @ Qf @ x_new[:, -1]
        new_cost += last_cost
        
        if new_cost < prev_cost:
            if new_cost < smallest_cost:
                smallest_cost = new_cost

            x = x_new
            u = u_new
            d_cost = new_cost - prev_cost
            prev_cost = new_cost
            
            back_track = back_track_base

            print("iteration : {},\tcost : {},\tchange in cost: {},\tlast state: {}".format(itr, round(new_cost, 4), round(d_cost, 4), x[:, -1]))
            if -d_cost < 1e-3 and itr >= 5:
                print("Converged at iteration {}, last state: {}".format(itr, x[:, -1]))
                break
            itr += 1
        else:
            back_track *= 0.6
            if abs(back_track) < 1e-8:
                print("Converged at iteration {}, last state: {}".format(itr, x_new[:, -1]))
                return x_new, u_new

            print("iteration : {}, Cost increased, prev: {}, new: {}, update backtrack term: {}".format(itr, round(prev_cost,4), round(new_cost,4), back_track))
        
    # return opt_x, opt_u
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

    partial_alp_theta = jacobian(alpha, theta)
    partial_alp_w = jacobian(alpha, w)
    partial_acc_theta = jacobian(acc, theta)
    partial_acc_w = jacobian(acc, w)

    A = jacobian(x1, x)
    B = jacobian(x1, F)

    Q_theta = 1 / (np.deg2rad(5)**2)
    Q_w = 1 / (np.deg2rad(10)**2)
    Q_p = 1 / (2**2)
    Q_v = 1 / (0.5**2)
    R_F = 1 / (1**2)
    Q = np.diag([Q_theta, Q_w, Q_p, Q_v]) / Q_theta
    R = np.diag([R_F]) / Q_theta
    Qf = Q * 20

    # Q = np.diag([1, 1, 0.01, 1])
    # R = np.diag([0.02])
    # Qf = np.diag([10, 10, 1, 10])
    # Qf = Q * 5

    ref = np.zeros((4,1))

    state = np.array([np.pi, 0, 0, 0]).reshape(4,1)
    state_list = state.flatten().tolist()

    sim_time = 15
    N = int(sim_time / h) + 1
    tspan = [i * h for i in range(N)]

    # Make initial trajectory
    x_traj = np.zeros((4, N))
    x_traj[:,0] = state.reshape(4)
    u = np.zeros((1, N))
    At = substitute([A], [theta, w, p, v], state_list)

    for i in range(1, N):
        # u[0, i] = np.random.normal(0, 0.1)
        x_traj[:, i] = cart_pole_model(x_traj[:,i-1].flatten().tolist(), [u[0, i-1]]).reshape(4)

    x_res, u = ilqr(x_traj, u, 40)

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(tspan, x_res[0,:], label="theta_res")
    # ax[0].plot(tspan, x_traj[0,:], label="theta")
    ax[1].plot(tspan, u[0,:], label="F")
    ax[0].legend()
    ax[1].legend()
    plt.show()
