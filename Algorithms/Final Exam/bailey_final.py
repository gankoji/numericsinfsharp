import sympy
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from solvers import gradientDescent, newtonIneq

## Problem 1: Unconstrained minimization
## First, the objective function
def sub_1(xk,xk1,yk,yk1):
    eps = 1e-1
    if (yk <= eps) or (yk1 <= eps):
        print("Domain error")
        return 1e6

    if math.fabs(yk - yk1) <= eps:
        return math.fabs(xk-xk1)/yk
    else:
        return math.fabs(math.log(yk)-math.log(yk1))*math.sqrt(1 + ((xk-xk1)**2/(yk-yk1)**2))

def f_1(x):
    xs = x[0:20]
    ys = x[20:40]
    s = 0.
    s += sub_1(xs[0],-5.,ys[0],12.)
    s += sub_1(12.,xs[19],5.,ys[19])
    for i in range(1,20):
        s += sub_1(xs[i],xs[i-1],ys[i],ys[i-1])

    return s

## Next, we need it's derivative
def g_1(x):
    delta = 1e-2
    f0 = f_1(x)
    dx = np.zeros(x.shape)
    for i in range(1,40):
        temp = x
        temp[i] = temp[i] + delta
        df = f_1(temp)
        dx[i] = (df - f0)/delta
    return dx

## Solve the problem!
def problem1():
    x = np.ones((40,))
    sol = gradientDescent(f_1,g_1,x,f_1(x)+1,5e-2, 5e-6)

    ## This next bit is a bit of tap dancing to get our solution
    ## points into line segments for plotting
    xs = sol[0:20]
    ys = sol[20:40]
    points = list(zip(xs,ys))
    lines = []
    for i in range(1,20):
        temp = []
        temp.append(points[i-1])
        temp.append(points[i])
        lines.append(temp)

    lc = mc.LineCollection(lines)

    ## Finally, plot those line segments and save the figure
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    plt.savefig("p1.png")


from sympy import *
from sympy.functions.elementary.exponential import log, sqrt

## Problem 2: Inequality Constrained Minimization
## First, symbolics (objective function, gradient, and hessian)
def problem2(r):
    x,y,z,t = symbols('x y z t')
    f0 = x - y -z + x**2 - y**2 + x**3 + z**3

    gs = []
    gs.append(x**2 + y**2 + z**2 - r**2)

    GS = []
    GS.append(lambdify([x,y,z,t], gs[0]))

    for i in range(0,1):
        f0 = f0 - (1/t)*log(-gs[i])

    dF = [diff(f0,x), diff(f0,y), diff(f0,z)]
    ddF = [[diff(dF[0],x), diff(dF[0],y), diff(dF[0],z)],
           [diff(dF[1],x), diff(dF[1],y), diff(dF[1],z)],
           [diff(dF[2],x), diff(dF[2],y), diff(dF[2],z)]]

    f = lambdify([x,y,z,t], f0)
    df = lambdify([x,y,z,t], dF)
    ddf = lambdify([x,y,z,t], ddF)

    def f213(p,t):
        x = p[0]
        y = p[1]
        z = p[2]

        return np.array(f(x,y,z,t))

    def g213(p,t):
        x = p[0]
        y = p[1]
        z = p[2]

        return np.array(df(x,y,z,t))

    def h213(p,t):
        x = p[0]
        y = p[1]
        z = p[2]

        return np.array(ddf(x,y,z,t))

    # Starting value found manually from lower tolerance runs
    # Here we're polishing roots by gradually increasing tolerance (again, manually)
    x0 = np.array([-0.2,r-0.1,0.3])
    print(f"Problem 2: Inequality Constrained Minimization")
    print(f"Radius of feasible region is {r}.")
    print(f"Optimizing via Newton's Method, starting at {x0[0], x0[1], x0[2]}.")

    x = newtonIneq(f213, GS, g213, h213, x0, 1e-6)
    print(f"Solved! Optimal point: ({x[0]}, {x[1]}, {x[2]}).")
    print(f"Optimal Function Value: {f213(x,1e9)}.")

problem2(1.)
problem2(2.)
problem2(3.)