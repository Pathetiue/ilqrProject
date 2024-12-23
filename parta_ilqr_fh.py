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

def ilqr(x_in, u_in, iterations):
    """
    x: 4 * N
    u: 1 * N
    """
    x_traj = x_in.copy()
    u_traj = u_in.copy()
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
        Vx[:, -1] = Qf @ x_traj[:, -1]
        Vxx[:, :, -1] = Qf

        du = np.zeros((1, N))
        kts = [None] * N
        Kts = [None] * N

        for k in range(N-2, -1, -1):
            x_traj[:, k][0] = wrap_angle(x_traj[:, k][0])

            # Compute Q function
            fx = substitute([A], [theta, w, p, v, F], x_traj[:, k].flatten().tolist() + u_traj[:, k].tolist())[0]
            fx = cas2arr(fx)
            fu = substitute([B], [theta, w, p, v, F], x_traj[:, k].flatten().tolist() + u_traj[:, k].tolist())[0]
            fu = cas2arr(fu)

            Qx = cost_x(x_traj[:, k]) + fx.T @ Vx[:, k+1]
            Qu = cost_u(u_traj[:, k]) + fu.T @ Vx[:, k+1]
            Qx = cost_x(x_traj[:, k]) + (fx.T @ Vx[:, k+1]).T
            Qu = cost_u(u_traj[:, k]) + (fu.T @ Vx[:, k+1]).T
            Qxx = cost_xx(x_traj[:, k]) + fx.T @ Vxx[:, :, k+1] @ fx
            Quu = cost_uu(u_traj[:, k]) + fu.T @ Vxx[:, :, k+1] @ fu
            Qux = fu.T @ Vxx[:, :, k+1] @ fx

            kt = -np.linalg.inv(Quu) @ Qu
            Kt = -np.linalg.inv(Quu) @ Qux

            kts[k] = kt
            Kts[k] = Kt

            du[:, k] = kt + Kt @ x_traj[:, k]
            Vx[:, k] = Qx - Kt.T @ Quu @ kt
            Vxx[:, :, k] = Qxx - Kt.T @ Quu @ Kt

        x_new = np.zeros((4, N))
        u_new = np.zeros((1, N))
        x_new[:, 0] = x_traj[:, 0].reshape(4)

        # Forward pass
        for k in range(N-1):
            du = back_track * kts[k] + Kts[k] @ x_new[:, k]
            u_new[:, k] = u_traj[:, k] + du
            # u_new[:, k] = np.clip(u_new[:, k], -100, 100)

            x_new[:, k+1] = cart_pole_model(x_new[:, k].flatten().tolist(), u_new[:, k].tolist()).reshape(4)
            x_new[:, k+1][0] = wrap_angle(x_new[:, k+1][0])
            pass

        new_cost = np.sum([cost_function(x_new[:, k], u_new[:, k]) for k in range(N-1)])
        last_cost = x_new[:, -1].T @ Qf @ x_new[:, -1]
        new_cost += last_cost
        
        if new_cost < prev_cost:
            if new_cost < smallest_cost:
                smallest_cost = new_cost

            x_traj = x_new
            u_traj = u_new
            d_cost = new_cost - prev_cost
            prev_cost = new_cost
            
            back_track = back_track_base

            # print("iteration : {},\tcost : {},\tchange in cost: {},\tlast state: {}".format(itr, round(new_cost, 4), round(d_cost, 4), x_traj[:, -1]))
            if -d_cost < 1e-6 and itr >= 3:
                # print("Converged at iteration {}, last state: {}".format(itr, x_traj[:, -1]))
                break
            itr += 1
        else:
            back_track *= 0.6
            if abs(back_track) < 1e-6:
                # print("Converged at iteration {}, last state: {}".format(itr, x_new[:, -1]))
                return x_traj, u_traj, kts, Kts, back_track

            # print("iteration : {}, Cost increased, prev: {}, new: {}, update backtrack term: {}".format(itr, round(prev_cost,4), round(new_cost,4), back_track))
        
    # return opt_x, opt_u
    return x_traj, u_traj, kts, Kts, back_track

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

    Q_theta = 1 / (np.deg2rad(4)**2)
    Q_w = 1 / (np.deg2rad(7)**2)
    Q_p = 1 / (0.3**2)
    Q_v = 1 / (0.1**2)
    R_F = 1 / (0.6**2)
    Q = np.diag([Q_theta, Q_w, Q_p, Q_v]) / Q_theta
    R = np.diag([R_F]) / Q_theta
    # Qf = np.diag([Q_theta * 2, Q_w * 5, Q_p, Q_v * 10]) / Q_theta
    Qf = Q

    ref = np.zeros((4,1))

    init_state = np.array([np.pi, 0, 0, 0]).reshape(4,1)
    state_list = init_state.flatten().tolist()

    pred_horiz = 1
    N = int(pred_horiz / h) + 1

    # Make initial trajectory
    x_traj = np.zeros((4, N))
    x_traj[:,0] = init_state.reshape(4)
    u_traj = np.zeros((1, N))
    At = substitute([A], [theta, w, p, v], state_list)

    for i in range(1, N):
        # u[0, i] = np.random.normal(0, 1)
        x_traj[:, i] = cart_pole_model(x_traj[:,i-1].flatten().tolist(), [u_traj[0, i-1]]).reshape(4)

    sim_time = 20
    n_sim_step = int((sim_time / h) + 1)
    tspan = np.linspace(0, sim_time, n_sim_step)

    x_res = np.zeros((4, n_sim_step))
    x_res[:,0] = init_state.reshape(4)
    u_res = np.zeros((1, n_sim_step))
    for ts in range(n_sim_step-1):
        print("sim step: {} / {}".format(ts, n_sim_step-2))
        # Solve optimal control
        _, u_ilqr, kts, Kts, back_track = ilqr(x_traj, u_traj, 10)
        du = back_track * kts[0] + Kts[0] @ x_res[:, ts]
        print("gain k: {}, K: {}, backtrack: {}".format(kts[0], Kts[0], back_track))
        u_ctrl = (u_traj[:,0] + du)[0]
        u_ctrl = np.clip(u_ctrl, -30, 30)
        u_res[:,ts] = u_ctrl
        # Forward dynamics
        x_res[:, ts+1] = cart_pole_model(x_res[:,ts].flatten().tolist(), [u_ctrl]).reshape(4)
        x_res[0, ts+1] = wrap_angle(x_res[0,ts+1])
        print("next state: {}, control: {}".format(x_res[:,ts+1], round(u_ctrl, 4)))
        # Reset initial trajectory
        x_traj = np.zeros((4, N))
        x_traj[:,0] = x_res[:,ts+1].reshape(4)
        # Reset initial control
        u_traj = np.zeros((1, N))
        for i in range(1, N):
            # u_traj[0, i-1] = (kts[i-1] + Kts[i-1] @ x_traj[:,i-1])[0]
            u_traj[0, i-1] = np.random.normal(0, 0.1)
            x_traj[:, i] = cart_pole_model(x_traj[:,i-1].flatten().tolist(), [u_traj[0, i-1]]).reshape(4)
        

    fig, ax = plt.subplots(3, 2)
    ax[2,1].axis("off")
    fig.suptitle("Prediction Horizon: {}s".format(pred_horiz))    
    ax[0, 0].plot(tspan, x_res[0,:], label="theta")
    ax[0, 0].set_title("theta")
    ax[0, 0].set_xlabel("time (s)")
    ax[0, 0].set_ylabel("theta (rad)")
    ax[1, 0].plot(tspan, x_res[1,:], label="w")
    ax[1, 0].set_title("w")
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_ylabel("w (rad/s)")
    ax[0, 1].plot(tspan, x_res[2,:], label="p")
    ax[0, 1].set_title("p")
    ax[0, 1].set_xlabel("time (s)")
    ax[0, 1].set_ylabel("p (m)")
    ax[1, 1].plot(tspan, x_res[3,:], label="v")
    ax[1, 1].set_title("v")
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_ylabel("v (m/s)")
    ax[2, 0].plot(tspan, u_res[0,:], label="F")
    ax[2, 0].set_title("F")
    ax[2, 0].set_xlabel("time (s)")
    ax[2, 0].set_ylabel("F (N)")
    ax[0, 0].legend()
    ax[1, 0].legend()
    ax[0, 1].legend()
    ax[1, 1].legend()
    ax[2, 0].legend()
    
    plt.show()
