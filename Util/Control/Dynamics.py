from sympy.physics.mechanics import *
from sympy import sin, cos, symbols, Matrix, solve, Array

""" from sympy import lambdify, Dummy
from scipy import array, hstack, linspace, ones
from scipy import random, interp, vstack
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sympy.integrals.transforms import laplace_transform
from sympy.integrals.transforms import inverse_laplace_transform
from sympy import *
import sympy as sympy
from sympy.abc import a, t, x, s, X, g, G
 """
def ordinaryDifferentialEquationSolver():
    # 3xdd +30xd + 63x = 4gd + 6g
    # g(t) = u_s(t) -> step function
    # x(0) = 4, xd(0) = 7
    x,g,X = symbols('x g X', cls = Function)
    x0 = 4 
    xd0 = 7
    g = Heaviside(t)
    laplace_transform(Heaviside(t), t, s)
    z = DiracDelta(t)
    

def trajectoryOptimization():
# Simple  pendulum
# Objective is to minimize effort of control needed to swing up
# Boundary conditions are theta_0 = 0, theta_tf = pi at t= tf

# Parameter Identification
# Find the parameters p such that the difference between the 
# model simulation y and the measurements ym is minimized
# xd = f(x,p) State
# y = g(x,p) Measurement
# Objective is to minimize J(p)
# With J(p) = Integral(y - y(p))**2 dt
    pass


def mobileRobot():
    #1) Define an inertial reference frame (origin), and coordinate origin

    #Inertial Reference Frame
    N = ReferenceFrame('N')
    #Define World Coordinate Origin
    O = Point('O')
    O.set_vel(N,0)

    # create dynamic symbol variables
    theta = dynamicsymbols('theta')
    x,y = dynamicsymbols('x,y')
    phi1,phi2 = dynamicsymbols('phi_1, phi_2')

    # Create the continous time state vectors
    q = Matrix([x,y,theta,phi1,phi2])
    dq = q.diff()

    #Constants for the wheels
    r = symbols('r')  # Radius of wheel
    m_w = symbols('m_w')  # Mass of the wheel
    t_w = symbols('t_w') # thickness of wheel

    # Constants for the Robot
    w = symbols('w') # 2*w is the width of the wheel base
    d = symbols('d') # Distance between the axel and the center of mass
    m_b = symbols('m_b')
    Ixx,Iyy,Izz = symbols('Ixx,Iyy,Izz') # Moments of inertia of body

    # Next, Define a reference frame to the robot
    # Assign points to the robot's wheel axis

    # Robot Reference Frame
    R = N.orientnew('R', 'Axis', [theta, N.z]) # The dyn var theta is linked

    # Center of wheel base
    Cw = O.locatenew('Cw', x*N.x + y*N.y) # New point that defines center of mobile robot
    Cw.set_vel(N, x.diff()*N.x + y.diff()*N.y) # Velocity of wheel axis ceneter (robot origin)

    # Points of wheel hubs
    H1 = Cw.locatenew('H1', -1*w*R.y)
    H2 = Cw.locatenew('H2',  w*R.y)
    # Set Velocity of these points
    H1.v2pt_theory(Cw,N,R)
    H2.v2pt_theory(Cw,N,R)

    # Create Rotating Reference Frames for the Wheels
    W1 = R.orientnew('W1', 'Axis', [phi1, R.y])
    W2 = R.orientnew('W2', 'Axis', [phi2, R.y])

    # Wheels are cylinders 
    #Calculate inertia of the wheel
    Iw = inertia(R, m_w*(3*r**2 + t_w**2)/12, m_w*r**2/2, m_w*(3*r**2 + t_w**2)/12)

    # Create Rigib Bodies For Wheels
    Wheel1 = RigidBody('Wheel1', H1, W1, m_w, (Iw, H1))
    Wheel2 = RigidBody('Wheel2', H2, W2, m_w, (Iw, H2))

    #Calculate inertia of body
    Ib = inertia(R, Ixx, Iyy, Izz)

    #Center of mass of body
    Cm = Cw.locatenew('Cm', d*R.x)
    Cm.v2pt_theory(Cw, N, R)

    #Create a rigid body object for body
    Body = RigidBody('Body', Cm, R, m_b, (Ib, Cm))

    #Create two points, where the wheels contact the ground
    C1 = H1.locatenew('C1', -1*r*R.z)
    C2 = H2.locatenew('C2', -1*r*R.z)
    #Calculate velocity of points
    C1.v2pt_theory(H1, N, W1)
    C2.v2pt_theory(H2, N, W2)

    #Express the velocity of points in the inertial frame
    con1 = C1.vel(N).express(N).args[0][0]
    con2 = C2.vel(N).express(N).args[0][0]
    #Create a matrix of constraints
    constraints = con1.col_join(con2)
    mprint(constraints)

    #Solve for dx, dy, and dtheta in terms of dphi1 and dphi2
    sol = solve(constraints, dq[:3])

    #Split the resulting dict into a rhs and lhs, that are equivalent
    sol_rhs = Array(sol.values())
    sol_lhs = Array(sol.keys())
    
    #Since sol_rhs = sol_lhs --> sol_rhs - sol_lhs = 0
    #This forms the basis of our constraint matrix.
    #Combining, and solving for a linear representation:
    c = Matrix(sol_rhs - sol_lhs).jacobian(dq[:5])
    mprint(c)

    coneqs = (c*dq)
    mprint(coneqs)

    T1, T2 = symbols('tau_1, tau_2')  # Torques from the wheels
    fl = [(H1, r*T1*R.x),
        (H2, r*T2*R.x)]

    #We are now ready to solve the for the equations of motion. First, calculate the lagrangian of the system $L = T - V$:
    Lag = Lagrangian(N, Wheel1, Wheel2, Body)
    lm = LagrangesMethod(Lag, q, hol_coneqs=coneqs, forcelist=fl, frame=N)

    le = lm.form_lagranges_equations()
    mprint(le)

    # Simulation
    rhs = lm.rhs()
    
    #Create dynamic symbols for current and voltage
    i_1, i_2 = dynamicsymbols('i_1, i_2')  # Currents through motor 1 and 2
    V_1, V_2 = symbols('V_1, V_2')  # Voltages across the motor terminals

    #Define some motor constants.
    #Assuming motor 1 and 2 are the same:
    R = symbols('R')  # Coil resistance
    L = symbols('L')  # Coil inductance
    K1, K2 = symbols('K1, K2')  # Motor constant

    #Define the motor dynamics
    di = Matrix([[-K1/L*phi1.diff() - R/L*i_1 + V_1/L],
                [-K2/L*phi2.diff() - R/L*i_2 + V_2/L]])


    #Define consts:
    params = [Izz,  t_w,  m_w,    r,  m_b,   w,    R,      L,   K1,  K2]
    values = [5, 0.15,  2.0, 0.15, 50.0, 0.6, 0.05,
            0.0001,  1.0, 1.0]  # Values of constants

    #Create a list of dynamic symbols for simulation
    dynamics = q.T.tolist()[0] + dq.T.tolist()[0] + \
        lm.lam_vec.T.tolist()[0] + [i_1, i_2]

    #Set the inputs to be the motor voltages
    inputs = [V_1, V_2]

    #Add the motor model to the rhs equations
    aug_rhs = rhs.col_join(di)

    #Substitute in K*i for T in the rhs equations
    aug_rhs = aug_rhs.subs({T1: K1*i_1, T2: K2*i_2})

    #Create a list of dynamic symbols for simulation
    dummys = [Dummy() for i in dynamics]
    dummydict = dict(zip(dynamics, dummys))
    #Sub in the dummy symbols
    rhs_dummy = aug_rhs.subs(dummydict)
    #Lambdify function
    rhs_func = lambdify(dummys + inputs + params, rhs_dummy)



#Create a function in the required format for odeint
def right_hand_side(x, t, ts, us, values):
    """Calculate the rhs of the integral, at
    time t, state x.
    ts, us are used to get the current input
    values are constants in the integral"""

    #Interp is needed to get u for timestep
    u1 = interp(t, ts, us[:, 0])
    u2 = interp(t, ts, us[:, 1])

    arguments = hstack((x, u1, u2, values))

    #Need call to array and reshape, as odeint
    #requires state vector to be 1d
    dx = array(rhs_func(*arguments))
    return dx.reshape(dx.size)

def bot2():
    #mobileRobot()

    # Define Origin
    N = ReferenceFrame("N")
    O = Point('O')
    O.set_vel(N, 0)

    # Create State Variables -- x,y of robot and phi1,phi2 of wheels
    theta = dynamicsymbols('theta')  # rotation of robot
    x, y = dynamicsymbols('x,y')  # Position of robot
    phi1, phi2 = dynamicsymbols('phi1, phi2')  # Angle of wheels
    q = Matrix([x, y, theta, phi1, phi2])  # state vector
    dq = q.diff()  # Derivative of state vector

    # Define Robot Constants
    # Constants for the wheels
    r = symbols('r')                                     # Radius of wheel
    m_w = symbols('m_w')                                 # Mass of the wheel
    # Thickness of the wheel
    t_w = symbols('t_w')

    # Constants for the Robot Body
    # 2*w is the width of the wheel base
    w = symbols('w')
    # Distance between axel and center of mass
    d = symbols('d')
    m_b = symbols('m_b')                                 # Mass of the body
    # Moments of inertia of body
    Ixx, Iyy, Izz = symbols('Ixx, Iyy, Izz')

    # Next Define Robots Coordinate Frame
    R = N.orientnew('R', 'Axis', [theta, N.z])
    Cw = O.locatenew('Cw', x*N.x + y*N.y)
    Cw.set_vel(N, x*N.x + y*N.y)

    # Coordinates of the two wheel hubs:
    H1 = Cw.locatenew('H1', -w*R.y)
    H2 = Cw.locatenew('H2', w*R.y)

    # Set Velocity of the Wheel hubs
    H1.v2pt_theory(Cw, N, R)
    H2.v2pt_theory(Cw, N, R)

    # create reference frames for each fixed wheel
    W1 = R.orientnew('W1', 'Axis', [phi1, R.y])
    W2 = R.orientnew('W2', 'Axis', [phi2, R.y])

    # Now that all kinematc frames have been defined, setup rigid body dynamics
    Iw = inertia(R, m_w*(3*r**2 + t_w**2)/12, m_w *
                 r**2/2, m_w*(3*r**2 + t_w**2)/12)

    # Create rigid bodies for wheels
    Wheel1 = RigidBody('Wheel1', H1, W1, m_w, (Iw, H1))
    Wheel2 = RigidBody('Wheel2', H2, W2, m_w, (Iw, H2))

    # Rigid Body for the Robot
    Ib = inertia(R, Ixx, Iyy, Izz)
    Cm = Cw.locatenew('Cm', d*R.x)
    Cm.v2pt_theory(Cw, N, R)
    Body = RigidBody('Body', Cm, R, m_b, (Ib, Cm))

    # Now, Add the constraints
    # Create two points, where the wheels contact the ground
    C1 = H1.locatenew('C1', -1*r*R.z)
    C2 = H2.locatenew('C2', -1*r*R.z)
    # Calculate velocity of points
    C1.v2pt_theory(H1, N, W1)
    C2.v2pt_theory(H2, N, W2)

    # Create Constraints
    con1 = C1.vel(N).express(N).args[0][0]
    con2 = C1.vel(N).express(N).args[0][0]
    constraints = con1.col_join(con2)
    # solving for dx, dy, dtheta in terms of dphi and dphi2
    sol = solve(constraints, dq[:3])
    sol_rhs = Matrix(list(sol.values()))
    sol_lhs = Matrix(list(sol.keys()))

    # Since sol_rhs = sol_lhs --> sol_rhs - sol_lhs = 0
    # This forms the basis of our constraint matrix.
    # Combining, and solving for a linear representation:
    c = (sol_rhs - sol_lhs).jacobian(dq[:5])
    # Constraints * dq = 0
    coneqs = (c*dq)

    # Define inputs on the system
    T1, T2 = symbols('tau1, tau2')
    forces = [(H1, T1/r*R.x), (H2, T2/r*R.x)]

    # Now we have the lagrangian
    Lag = Lagrangian(N, Wheel1, Wheel2, Body)
    lm = LagrangesMethod(Lag, q, nonhol_coneqs=coneqs, forcelist=fl, frame=N)
    le = lm.form_lagranges_equations()

    # Simulation


def testin():
    q1, q2 = dynamicsymbols('q1 q2')
    q1d, q2d = dynamicsymbols('q1 q2', 1)
    L = q1d**2 + q2d**2
    LM = LagrangesMethod(L, [q1, q2])
    LM.form_lagranges_equations()


def Model():
    pass



if __name__ == "__main__":
    # Human Leg
    theta1,theta2,theta3 = dynamicsymbols('theta1, theta2, theta3')
    inertial_frame = ReferenceFrame('I')
    lower_leg_frame = ReferenceFrame('L')
    upper_leg_frame = ReferenceFrame('U')
    torse_frame = ReferenceFrame('T')
    lower_leg_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))
    upper_leg_frame.orient(lower_leg_frame, 'Axis', (theta2, lower_leg_frame.z) )
    torse_frame.orient(upper_leg_frame, 'Axis', (theta3, upper_leg_frame))

    # The reference frames dont have translations relative to one another,
    # Only the points translate and rotate (within reference frames)
    ankle = Point('k')
    lower_leg_length = symbols('L')
    lower_leg = Point('l_l')
    knee = Point('k')
    knee.set_pos(ankle, lower_leg_length*lower_leg_frame.y)
    upper_leg_length = symbols('UL')
    hip = Point("h")
    hip.set_pos(knee,upper_leg_length*upper_leg_frame.y)

    # Center of mass locations
    lower_leg_cm_length, upper_leg_cm_frame, torso_cm_length = symbols('llcm, ulcm, tcml')


