from sympy import *
from sympy.vector import gradient
from sympy.vector.coordsysrect import CoordSys3D
import numpy as np
from sympy.utilities.lambdify import lambdify

def solve_least_squares(x, y):
    n = x.size
    m = np.sum((x - np.mean(x))*(y - np.mean(y)))/(np.sum((x - np.mean(x))**2))
    b = (np.sum(y) - m*np.sum(x))/n
    coeffs = np.array([b, m])
    return coeffs

def q_16_2():
    x = np.array([-2., -1., 0., 2., 3.])
    y = np.array([-3., -1., 5., 5., 1.])

    c = solve_least_squares(x, y)
    print(c)

q_16_2()

def q_17_1():
    x, y = symbols('x y')
    g = sqrt(x**2 + 1) + x
    gxy = g.subs(x, (y**2 - x**2))
    h2 = 40*sqrt(gxy) + (x - 10)**2 + y**2
    print(f"Objective Function: {h2}")

    R = CoordSys3D('R')
    h2R = h2.subs([(x, R.x), (y, R.y)])
    grad = gradient(h2R)
    print(f"Gradient of Objective Function: {grad}")

    newGrad = lambdify([x, y], derive_by_array(h2, (x, y)))
    newH = lambdify([x, y], h2)

    x, y, old_F, F, dt =  -50.0, 40.0, 0., 6000., 0.01
    F_x = newGrad(x,y)

    print(f"Initial value: {newH(x,y)}")
    print(f"\n\nMinimizing by Gradient Descent. Starting at ({x}, {y}).")
    while abs(F - old_F) > 1e-10:

        old_x, old_y, old_F, F_x, = x, y, F, newGrad(x,y)
        x, y, dt = x - dt*F_x[0], y - dt*F_x[1], 1.1*dt
        F = newH(x,y)

        if (F > old_F):
            x, y, F, old_F, dt = old_x, old_y, old_F, old_F + 1., 0.5*dt

    print("\nSolution Found. ")
    print(f"Optimal Solution Point: ({x}, {y})")
    print(f"Objective Function Value: {F}")

q_17_1()

def q_17_2_a():
    x, y = symbols('x y')
    h2 = (x + 3)**2 + y**2*exp(-2*x)
    print(f"Objective Function: {h2}")

    R = CoordSys3D('R')
    h2R = h2.subs([(x, R.x), (y, R.y)])
    grad = gradient(h2R)
    print(f"Gradient of Objective Function: {grad}")

    newGrad = lambdify([x, y], derive_by_array(h2, (x, y)))
    newH = lambdify([x, y], h2)

    x, y, old_F, F, dt =  0, 1.0, 0., 6000., 0.01
    F_x = newGrad(x,y)

    print(f"\n\nMinimizing by Gradient Descent. Starting at ({x}, {y}).")
    while abs(F - old_F) > 1e-10:
        old_x, old_y, old_F, F_x, = x, y, F, newGrad(x,y)
        x, y, dt = x - dt*F_x[0], y - dt*F_x[1], 1.1*dt
        F = newH(x,y)

        if (F > old_F):
            x, y, F, old_F, dt = old_x, old_y, old_F, old_F + 1., 0.5*dt

    print("\nSolution Found. ")
    print(f"Optimal Solution Point: ({x}, {y})")
    print(f"Objective Function Value: {F}")

q_17_2_a()

def q_17_2_b():
    x, y = symbols('x y')
    h2 = (x + 3)**2 + y**2*exp(-2*x)
    h2 = h2.subs([(y, y/20)])
    print(f"Objective Function: {h2}")

    R = CoordSys3D('R')
    h2R = h2.subs([(x, R.x), (y, R.y)])
    grad = gradient(h2R)
    print(f"Gradient of Objective Function: {grad}")

    newGrad = lambdify([x, y], derive_by_array(h2, (x, y)))
    newH = lambdify([x, y], h2)

    x, y, old_F, F, dt =  0, 20.0, 0., 6000., 0.01
    F_x = newGrad(x,y)

    print(f"\n\nMinimizing by Gradient Descent. Starting at ({x}, {y}).")
    while abs(F - old_F) > 1e-10:
        old_x, old_y, old_F, F_x, = x, y, F, newGrad(x,y)
        x, y, dt = x - dt*F_x[0], y - dt*F_x[1], 1.1*dt
        F = newH(x,y)

        if (F > old_F):
            x, y, F, old_F, dt = old_x, old_y, old_F, old_F + 1., 0.5*dt

    print("\nSolution Found. ")
    print(f"Optimal Solution Point: ({x}, {y})")
    print(f"Objective Function Value: {F}")

q_17_2_b()