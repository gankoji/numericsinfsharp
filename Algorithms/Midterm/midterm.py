import sys
from sympy import *
from sympy.functions.elementary.exponential import log, sqrt, exp
from rkf45 import r8_rkf45
import numpy as np
import math

def gradientDescent (f, gradF, x0, f0, dt0):
    x = x0
    fval = f0
    dt = dt0
    newf = f(x)

    while (math.fabs (newf - fval)) > 1e-6:
        newf = f(x)
        newg = gradF(x)
        newx = x - dt*newg

        print(newx)
        print(newf)
        print(newg)
        inter = f(newx)
        if (inter > fval):
            fval += 1.0
            dt = dt*0.5
        else:
            dt = 1.1*dt
            x = newx

    return x

t = 100000.
def newtonL2Eq(f, dF, ddF, xin):
    x = xin
    while True:
        F = f(x)
        DF = dF(x)

        if math.fabs(np.linalg.norm(DF)) < 1e-8:
            break

        DDF = ddF(x)

        if np.linalg.cond(DDF) < 1/sys.float_info.epsilon:
            d = -np.linalg.solve(DDF,DF)
        else:
            break
        flag = False

        while (f(x + d) >= F):
            d = 0.5*d
            if math.fabs(np.linalg.norm(d)) < 1e-6:
                if flag:
                    break
                d = -DF
                flag = True
            
        x = x + d
        
    return x

def newtonL2Ineq(f, g, dF, ddF, xin):
    x = xin
    flag = False
    while True:
        F = f(x)
        DF = dF(x)

        if math.fabs(np.linalg.norm(DF)) < 1e-8:
            break

        DDF = ddF(x)
        d = -np.linalg.solve(DDF,DF)
        if (g(x+d) >= 0) and flag:
            break
        flag = False

        while (g(x + d) >= 0) or (f(x + d) >= F):
            d = 0.5*d
            if np.linalg.norm(d) < 1e-8:
                flag = True
                break
            
        x = x + d
        
    return x

## Problem 1: Unconstrained Optimization
x,y = symbols('x y')
h = ((3*x + y)*exp(3*x + 1*y) + 3*y*exp(3*y) - (2*x + 3*y)*exp(-2*x -3*y))/(exp(3*x+1*y) + exp(3*y) + exp(-2*x -3*y))
#h = (3*x + y) + 3*y -2*x -3*y
dhdx = diff(h,x)
dhdy = diff(h,y)
gradH = [dhdx, dhdy]
hessH = [[diff(dhdx,x), diff(dhdx, y)], [diff(dhdy, x), diff(dhdy, y)]]
h = lambdify([x,y], h)
dH = lambdify([x, y], gradH)
ddH = lambdify([x,y], hessH)


def f1(p):
    x = p[0]
    y = p[1]

    return h(x,y)

def df1(p):
    x = p[0]
    y = p[1]

    return np.array(dH(x,y))

def ddf1(p):
    x = p[0]
    y = p[1]

    return np.array(ddH(x,y))

x0 = np.array([-1, -1])
print(f"Question 1: Unconstrained Minimization")
print(f"Optimizing via Newton's Method, starting at ({x0[0]},{x0[1]}).")
x = newtonL2Eq(f1, df1, ddf1, x0)
print(f"Solved! Optimal point: ({x[0]}, {x[1]}).")
print(f"Optimal Function Value: {f1(x)}.")

## Problem 2: Projectile Range Optimization
## Define state: [x y u v]
def df2(p):
    x = p[0]
    y = p[1]
    u = p[2]
    v = p[3]

    dx = u
    dy = v
    fac = math.sqrt(u**2 + v**2)
    du = -fac*u
    dv = -1 -fac*v

    return np.array([dx, dy, du, dv])
    

def rk4_step(dt, p, df):
    k1 = dt*df(p)
    k2 = dt*df(p + k1/2)
    k3 = dt*df(p + k2/2)
    k4 = dt*df(p + k3)

    return p + (1/6.)*k1 + (1/3.)*k2 + (1/3.)*k3 + (1/6.)*k4

def fire_shot(theta):
    trad = theta*math.pi/180.
    x = np.array([0,0,10*math.cos(trad),10*math.sin(trad)])
    while x[1] >= 0:
        x = rk4_step(0.01, x, df2) 
    return x[0]

def sweep(start, stop, nsteps):
    ranges = np.zeros((nsteps,))
    for i,t in enumerate(np.linspace(start, stop,nsteps)):
        ranges[i] = fire_shot(t)
    maxval = np.max(ranges)
    maxidx = np.argmax(ranges)
    return maxidx, maxval

def search():
    oldRange = -1.
    newRange = 0.
    start = 0.
    stop = 90.
    nsteps = int(1e2)
    delta = 0.25

    while math.fabs(oldRange - newRange) >= 1e-4:
        oldRange = newRange
        idx, newRange = sweep(start, stop, nsteps)
        angles = np.linspace(start, stop, nsteps)
        best = angles[idx]
        start = best*(1-delta)
        stop = best*(1+delta)

    print(f"Optimum Angle Found: {angles[idx]}")
    print(f"Optimum Range: {best}")

search()