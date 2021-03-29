import sys
from sympy import *
from sympy.functions.elementary.exponential import log, sqrt, exp
import numpy as np
import math
from optimization import *

## Problem 1: Unconstrained Optimization
def problem1():
    x,y = symbols('x y')
    h = ((3*x + y)*exp(3*x + 1*y) + 3*y*exp(3*y) - (2*x + 3*y)*exp(-2*x -3*y))/(exp(3*x+1*y) + exp(3*y) + exp(-2*x -3*y))
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

#problem1()

## Problem 2: Projectile Range Optimization
## Define state: [x y u v]
def problem2():
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
        print(f"Optimum Range: {newRange}")
    
    search()

#problem2()

## Problem 3: Constrained Minimization
# Minimize f(x,y,z) = (x-1)**2 + (y-1)**2 + (z-5)**2
# Subject to h(x,y,z) = 0 = x**2 - y**2 - 2z

def problem3():
    def problem3Symbolics():
        x,y,z = symbols('x y z')
        f = (x-1)**2 + (y-1)**2 + (z-5)**2
        h = x**2 - y**2 - 2*z
        h = f + t*(h**2)
        dhdx = diff(h,x)
        dhdy = diff(h,y)
        dhdz = diff(h,z)
        gradH = [dhdx, dhdy, dhdz]
        hessH = [[diff(dhdx,x), diff(dhdx, y), diff(dhdx, z)], [diff(dhdy,x), diff(dhdy, y), diff(dhdy, z)], [diff(dhdz,x), diff(dhdz, y), diff(dhdz, z)]]
        h = lambdify([x,y,z], h)
        dH = lambdify([x,y,z], gradH)
        ddH = lambdify([x,y,z], hessH)

        return h, dH, ddH

    h, dH, ddH = problem3Symbolics()

    def f3(p):
        x = p[0]
        y = p[1]
        z = p[2]

        return h(x,y,z)

    def g3(p):
        x = p[0]
        y = p[1]
        z = p[2]

        return np.array(dH(x,y,z))

    def h3(p):
        x = p[0]
        y = p[1]
        z = p[2]

        return np.array(ddH(x,y,z))

    x0 = np.array([1,1,0])
    print(f"Question 3: Equality Constrained Minimization")
    print(f"Optimizing via Newton's Method, starting at ({x0[0]},{x0[1]},{x0[2]}).")
    x = newtonL2Eq(f3, g3, h3, x0)
    print(f"Solved! Optimal point: ({x[0]}, {x[1]}, {x[2]}).")
    print(f"Optimal Function Value: {f3(x)}.")

#problem3()

## Problem 4: Inequality *and* Equality Constrained Minimization
t = 100000.
def problem4Symbolics():
    x1,y1,x2,y2 = symbols('x1 y1 x2 y2')
    f = 3*y2 - 2*y1
    h = (x1 - x2)**2 + (y1 - y2)**2 - 2
    g11 = y1 - x1
    g12 = y2 - x2
    g21 = x1**2 + y1**2 - 4*y1
    g22 = x2**2 + y2**2 - 4*y2
    h = f + t*(h**2) - (1/t)*log(-g11) - (1/t)*log(-g12) - (1/t)*log(-g21) - (1/t)*log(-g22)
    dhdx = diff(h,x1)
    dhdy = diff(h,y1)
    dhdx2 = diff(h,x2)
    dhdy2 = diff(h,y2)
    gradH = [dhdx, dhdy, dhdx2, dhdy2]
    hessH = [[diff(dhdx,x1), diff(dhdx, y1), diff(dhdx, x2), diff(dhdx, y2)],
             [diff(dhdy,x1), diff(dhdy, y1), diff(dhdy, x2), diff(dhdy, y2)],
             [diff(dhdx2,x1), diff(dhdx2, y1), diff(dhdx2, x2), diff(dhdx2, y2)],
             [diff(dhdy2,x1), diff(dhdy2, y1), diff(dhdy2, x2), diff(dhdy2, y2)]]
    args = [x1,y1,x2,y2]
    h = lambdify(args, h)
    dH = lambdify(args, gradH)
    ddH = lambdify(args, hessH)
    g11 = lambdify(args, g11)
    g12 = lambdify(args, g12)
    g21 = lambdify(args, g21)
    g22 = lambdify(args, g22) 
    return h, dH, ddH, g11, g12, g21    

def problem4():
    h, dH, ddH, g11, g12, g21, g22 = problem4Symbolics()
    def f4(p):
        x1 = p[0]
        y1 = p[1]
        x2 = p[2]
        y2 = p[3] 
        return h(x1,y1,x2,y2) 
    def g4(p):
        x1 = p[0]
        y1 = p[1]
        x2 = p[2]
        y2 = p[3] 
        return np.array(dH(x1,y1,x2,y2))
    def h4(p):
        x1 = p[0]
        y1 = p[1]
        x2 = p[2]
        y2 = p[3]
        return np.array(ddH(x1,y1,x2,y2)) 
    def c4(p):
        x1 = p[0]
        y1 = p[1]
        x2 = p[2]
        y2 = p[3] 
        a = g11(x1, y1, x2, y2)
        b = g12(x1, y1, x2, y2)
        c = g21(x1, y1, x2, y2)
        d = g22(x1, y1, x2, y2)
        if (a >= 0):
            return 1
        if b >= 0:
            return 1
        if c >= 0:
            return 1
        if d >= 0:
            return 1 
        return 0   
    x0 = np.array([1,1,0,0])
    print(f"Question 4: Inequality and Equality Constrained Minimization")
    print(f"Optimizing via Newton's Method, starting at ({x0[0]},{x0[1]},{x0[2]},{x0[3]}).")
    x = newtonL2Ineq(f4, c4, g4, h4, x0)
    print(f"Solved! Optimal point: ({x[0]}, {x[1]}, {x[2]},{x[3]}).")
    print(f"Optimal Function Value: {f4(x)}.")

problem4()

def problem5():
    print("Hey, we're here!")

problem5()