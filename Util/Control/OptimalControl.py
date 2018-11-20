import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
import control
import sympy as sym
from sympy import cos, sin

# Idea: subject x4 to a minimization with respect to the control input
def try_gekko():
    m = GEKKO()

    nt = 101
    m.time = np.linspace(0,1,nt)

    # Parameters
    u = m.MV(value = 9, lb = -4, ub = 10)
    u.STATUS = 1
    u.DCOST = 0

    # Variables
    t = m.Var(value = 0)
    x1 = m.Var(value = 0)
    x2 = m.Var(value=-1)
    x3 = m.Var(value=np.sqrt(5))
    x4 = m.Var(value=0)

    p = np.zeros(nt)
    p[-1] = 1
    final = m.Param(value = p)

    m.Equation(t.dt() == 1)
    m.Equation(x1.dt() == x2)
    m.Equation(x2.dt() == -x3*u + 16*t - 8)
    m.Equation(x3.dt ==u)
    m.Equation(x4.dt() == x1**2 + x2**2 + .0005*(x2 + 16*t - 8 - 0.1*x3*u**2))**2

    m.Obj(x4*final)

def ctrb(A,B):
    return control.obsv(A,B)

def obsv(A,C):
    return control.ctrb(A,C)

def induced_matrix_norm(A):
    _,S,_ = np.linalg.svd(A)
    return np.max(S)

def frobenius_norm(A):
    return np.sqrt(np.trace(A*A))

def sym_round2zero(m, e):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if (isinstance(m[i, j], sym.Float) and m[i, j] < e):
                m[i, j] = 0
    return m

def sym_linearize_system(fx,x):
    return fx.jacobian(x)

def sym_transfer_function(A,B,C = None, D = None):
    print(A.rows, A.cols)
    I = np.matrix(np.eye(A.rows))
    s = sym.symbols('s')
    sI_A = s*I - A
    inv_sI_A = sI_A.inv()
    inv_sI_A_B = inv_sI_A*B
    Cinv_sI_A_B = 0
    Cinv_sI_A_B_D = 0
    if C is not None:
        Cinv_sI_A_B = C*inv_sI_A_B
        return Cinv_sI_A_B
    if D is not None:
        Cinv_sI_A_B_D = Cinv_sI_A_B + D
        return Cinv_sI_A_B_D
    return inv_sI_A_B

def test_sympy():
    s, b, m, t = sym.symbols('s, b, m, t')
    A = sym.Matrix([[0, 1], [0, -b/m]])
    B = sym.Matrix([[0], [1/m]])
    C = sym.Matrix([[1,0],[0,1]])


    s, m, b, M, g, t, I, l,F = sym.symbols('s, m, b, M, g, t, I, l,F')
    [x,dx,theta,dtheta] = sym.symbols('x, xd, theta, dtheta')

    S = sym.Matrix([[M + m, m*l*cos(theta)],[I + m*l**2, m*l*sin(theta)]])
    fx = sym.Matrix([[m*l*dtheta**2*sin(theta) -b*dx],[-m*g*l*sin(theta)]])
    gx = sym.Matrix([[0],[1]])
    fx = S.inv()*fx
    """ A = linearize_system(fx, [x,dx,theta,dtheta])
    A=sym.simplify(round2zero(A.subs([(x, 0), (dx, 0), (theta, np.pi), (dtheta, 0)]), 1e-6))
    print(A)
    theta = sym.Function('theta')
    x = sym.Function('x')
    dx = x.diff(t)
    ddx = dx.diff(t)
    dtheta = theta.diff(t)
    ddtheta = dtheta.diff(t)

    tf = transfer_function(A,B)
    print(tf) """
