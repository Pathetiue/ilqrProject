import numpy as np
from casadi import *
import control # conda install -c conda-forge control slycot


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
    At = substitute(A, [theta, w, p, v, F], x + u)
    Bt = substitute(B, [theta, w, p, v, F], x + u)
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

def ilqr(x, u, iterations, Q, R):
    """
    x: list [theta, w, p, v]
    u: list [F]
    """
    min_du = 1e-3
    min_dJ = 1e-4
    min_dx = 1e-4

    pass

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

    alpha = (g * sin(theta) + cos(theta) * ( -F - mp * l * w**2 * sin(theta)) / (mc + mp)) / (l* (4/3 - mp*cos(theta)**2/(mc + mp)))
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
    print(B)

    Q = diag([1, 0.1, 0.1, 0.1])
    R = diag([1])

    ref = np.zeros((4,1))

    state = np.array([np.pi, 0, 0, 0]).reshape(4,1)
    state_list = state.flatten().tolist()
    
    x_1 = cart_pole_model(state_list, [0])
    
    sim_time = 5
    n_step = int(sim_time / h) + 1
    tspan = [i * h for i in range(n_step)]

    # Make initial trajectory
    x_traj = np.zeros((4, n_step))
    x_traj[:,0] = state.reshape(4)
    u = np.zeros((1, n_step))
    for i in range(1, n_step):
        x_traj[:, i] = cart_pole_model(x_traj[:,i-1].flatten().tolist(), [u[0, i-1]]).reshape(4)

    

