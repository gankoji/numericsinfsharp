from sympy import *
from sympy.functions.elementary.exponential import log
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
        d = -np.linalg.solve(DDF,DF)
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

x,y = symbols('x y')
h = ((6*x + 29)**2)*((x-1)**2) + 12*(6*x + 31)*(x-1)*y**2 + 36*y**4 
dhdx = diff(h,x)
dhdy = diff(h,y)
gradH = [dhdx, dhdy]
hessH = [[diff(dhdx,x), diff(dhdx, y)], [diff(dhdy, x), diff(dhdy, y)]]
h = lambdify([x,y], h)
dH = lambdify([x, y], gradH)
ddH = lambdify([x,y], hessH)

def f191(p):
    x = p[0]
    y = p[1]

    return x**2 + y**2 + t*h(x,y)

def g191(p):
    x = p[0]
    y = p[1]

    return np.array([2*x, 2*y]) + t*np.array(dH(x,y))

def gg191(p):
    x = p[0]
    y = p[1]

    return np.array([[2, 0],[0, 2]]) + t*np.array(ddH(x,y))

print(f"Question 19.1: Equality Constrained Minimization")
print(f"Optimizing via Newton's Method, starting at (0,0), with {t} slack.")

x = newtonL2Eq(f191, g191, gg191, np.array([0, 0]))
print(f"Solved! Optimal point: ({x[0]}, {x[1]}).")
print(f"Optimal Function Value: {f191(x)}.")
print(f"Constraint Value: {h(x[0], x[1])}.")

## Problem 21.1
## Minimize (x-1)**2 + (y-1)**2
## Subject To y**2 + x**3 <= 0

x,y,t = symbols('x y t')
f = (x-1)**2 + (y-1)**2 - (1/t)*log(-(y**2 + x**3))
df = [diff(f,x), diff(f,y)]
ddf = [[diff(df[0], x), diff(df[0],y)], [diff(df[1],x), diff(df[1],y)]]

F = lambdify([x, y, t], f)
dF = lambdify([x,y,t], df)
ddF = lambdify([x,y,t], ddf)

t = 100000.
def f211(p):
    x = p[0]
    y = p[1]

    return F(x,y,t)

def g211(p):
    x = p[0]
    y = p[1]

    return np.array(dF(x,y,t))

def h211(p):
    x = p[0]
    y = p[1]

    return np.array(ddF(x,y,t))

print(f"Question 21.1: Inequality Constrained Minimization")
print(f"Optimizing via Newton's Method, starting at (-1,0), with {t} slack.")

x = newtonL2Eq(f211, g211, h211, np.array([-1., 0]))
print(f"Solved! Optimal point: ({x[0]}, {x[1]}).")
print(f"Optimal Function Value: {f191(x)}.")
print(f"Constraint Value: {h(x[0], x[1])}.")