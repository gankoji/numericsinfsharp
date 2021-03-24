import sys
from sympy import *
from sympy.functions.elementary.exponential import log, sqrt
import numpy as np
import math

def gradientDescent (f, gradF, x0, f0, dt0):
    x = x0
    fval = f0
    dt = dt0
    newf = f(x)
    newg = gradF(x)

    while (math.fabs (newf - fval)) > 1e1:
        newf = f(x)
        newg = gradF(x)
        newx = x - dt*newg
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

def newtonL2VIneq(f, gs, dF, ddF, xin):
    x = xin
    flag = False
    breakFlag = False
    while True:
        F = f(x)
        DF = dF(x)

        if math.fabs(np.linalg.norm(DF)) < 1e-4:
            break

        DDF = ddF(x)
        d = (-2e-2)*np.linalg.solve(DDF,DF)
        whileFlag = True
        count = 0
        while whileFlag:
            d = (0.5)*d
            count += 1
            if count > 200:
                breakFlag = True
                break

            if math.fabs(np.linalg.norm(d)) < 1e-8:
                breakFlag = True 
                break
            for i,g in enumerate(gs):
                conVal = g(x[0]+d[0], x[1]+d[1], x[2]+d[2])
                if ( conVal > 0):
                    print(f"X: {x[0]+d[0]}, {x[1]+d[1]}, {x[2]+d[2]}")
                    print(f"Constraint {i}: {conVal}")
                    whileFlag = True
                    break
            whileFlag = False

        if breakFlag:
            break
        breakFlag = False
        flag = False

        while (g(x[0]+d[0], x[1]+d[1], x[2]+d[2]) >= 0) or (f(x + d) >= F):
            d = 0.5*d
            if np.linalg.norm(d) < 1e-8:
                flag = True
                break
            
        x = x + d
        
    return x

x,y = symbols('x y')
h = x*y + 1
h = x**2 + (-1/x)**2# + t*(h**2)
dhdx = diff(h,x)
dhdy = diff(h,y)
gradH = [dhdx, dhdy]
hessH = [[diff(dhdx,x), diff(dhdx, y)], [diff(dhdy, x), diff(dhdy, y)]]
gradH = dhdx
hessH = diff(dhdx, x)
# print(h)
print(gradH)
# print(hessH)
h = lambdify([x,y], h)
dH = lambdify([x, y], gradH)
ddH = lambdify([x,y], hessH)


def f191(p):
    x = p[0]
    y = p[1]

    return h(x,y)

def g191(p):
    x = p[0]
    y = p[1]

    return np.array(dH(x,y))

def gg191(p):
    x = p[0]
    y = p[1]

    return np.array(ddH(x,y))

x0 = np.array([-0.5, -0.5])
print(f"Question 19.1: Equality Constrained Minimization")
print(f"Optimizing via Newton's Method, starting at ({x0[0]},{x0[1]}), with {t} slack.")
x = newtonL2Eq(f191, g191, gg191, x0) 
print(f"Solved! Optimal point: ({x[0]}, {x[1]}).")
print(f"Optimal Function Value: {f191(x)}.")
print(f"Constraint Value: {h(x[0], x[1])}.")