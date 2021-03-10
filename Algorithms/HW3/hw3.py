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

x,y = symbols('x y')
h = ((6*x + 29)**2)*((x-1)**2) + 12*(6*x + 31)*(x-1)*y**2 + 36*y**4 
h = x**2 + y**2 + t*h
dhdx = diff(h,x)
dhdy = diff(h,y)
gradH = [dhdx, dhdy]
hessH = [[diff(dhdx,x), diff(dhdx, y)], [diff(dhdy, x), diff(dhdy, y)]]
print(h)
print(gradH)
print(hessH)
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

print(f"Question 19.1: Equality Constrained Minimization")
print(f"Optimizing via Newton's Method, starting at (0,0), with {t} slack.")

x = newtonL2Eq(f191, g191, gg191, np.array([0, 0]))
print(f"Solved! Optimal point: ({x[0]}, {x[1]}).")
print(f"Optimal Function Value: {f191(x)}.")
print(f"Constraint Value: {h(x[0], x[1])}.")

x,y,z = symbols('x y z')
f0 = 4*x - 2*y - z
dfdx = diff(f0,x)
dfdy = diff(f0,y)
dfdz = diff(f0,z)
gradF1 = [dfdx, dfdy, dfdz]
hessF1 = [[diff(dfdx,x), diff(dfdx, y), diff(dfdx,z)], [diff(dfdy, x), diff(dfdy, y), diff(dfdy,z)],[diff(dfdz,x), diff(dfdz,y), diff(dfdz,z)]]
f = lambdify([x,y,z], f0)
dF = lambdify([x,y,z], gradF1)
ddF = lambdify([x,y,z], hessF1)
h1 = x**2 + y**2 + z**2 - 1
h2 = x**6 + y**6 + z**6 + 24*x**2*y**2*z**2 - 1
dhdx = diff(h1,x)
dhdy = diff(h1,y)
dhdz = diff(h1,z)
gradH1 = [dhdx, dhdy, dhdz]
hessH1 = [[diff(dhdx,x), diff(dhdx, y), diff(dhdx,z)], [diff(dhdy, x), diff(dhdy, y), diff(dhdy,z)],[diff(dhdz,x), diff(dhdz,y), diff(dhdz,z)]]
h1 = lambdify([x,y,z], h1)
dH1 = lambdify([x,y,z], gradH1)
ddH1 = lambdify([x,y,z], hessH1)
dhdx = diff(h2,x)
dhdy = diff(h2,y)
dhdz = diff(h2,z)
gradH2 = [dhdx, dhdy, dhdz]
hessH2 = [[diff(dhdx,x), diff(dhdx, y), diff(dhdx,z)], [diff(dhdy, x), diff(dhdy, y), diff(dhdy,z)],[diff(dhdz,x), diff(dhdz,y), diff(dhdz,z)]]
h2 = lambdify([x,y,z], h2)
dH2 = lambdify([x,y,z], gradH2)
ddH2 = lambdify([x,y,z], hessH2)

def f192(p):
    x = p[0]
    y = p[1]
    z = p[2]

    return f(x,y,z) + t*(h1(x,y,z)) + t*(h2(x,y,z))

def g192(p):
    x = p[0]
    y = p[1]
    z = p[2]

    return np.array(dF(x,y,z)) + t*np.array(dH1(x,y,z)) + t*np.array(dH2(x,y,z))

def h192(p):
    x = p[0]
    y = p[1]
    z = p[2]

    return np.array(ddF(x,y,z)) + t*np.array(ddH1(x,y,z)) + t*np.array(ddH2(x,y,z))

print(f"Question 19.2: Equality Constrained Minimization")
print(f"Optimizing via Newton's Method, starting at (0,0,0), with {t} slack.")

s = math.sqrt(1/3.)
x = newtonL2Eq(f192, g192, h192, np.array([s, s, s]))
print(f"Solved! Optimal point: ({x[0]}, {x[1]}, {x[2]}).")
print(f"Optimal Function Value: {f192(x)}.")
print(f"Constraint 1 Value: {h1(x[0], x[1], x[2])}.")
print(f"Constraint 2 Value: {h2(x[0], x[1], x[2])}.")
## Problem 21.1
## Minimize (x-1)**2 + (y-1)**2
## Subject To y**2 + x**3 <= 0

x,y,t = symbols('x y t')
f = (x-1)**2 + (y-1)**2# - (1/t)*log(-(y**2 + x**3))
df = [diff(f,x), diff(f,y)]
ddf = [[diff(df[0], x), diff(df[0],y)], [diff(df[1],x), diff(df[1],y)]]

g = (y**2 + x**3)
dg = [diff(g,x), diff(g,y)]
ddg = [[diff(dg[0], x), diff(dg[0], y)],[diff(dg[1], x), diff(dg[1], y)]]
F = lambdify([x, y], f)
dF = lambdify([x,y], df)
ddF = lambdify([x,y], ddf)
G = lambdify([x,y], g)
dG = lambdify([x,y], dg)
ddG = lambdify([x,y], ddg)

t = 1000000.
def f211(p):
    x = p[0]
    y = p[1]

    return F(x,y) - (1/t)*math.log(-G(x,y))

def g211(p):
    x = p[0]
    y = p[1]

    return np.array(dF(x,y)) - (1/t)/np.array(dG(x,y))

def constraint211(p):
    x = p[0]
    y = p[1]

    return np.array(G(x,y))

def h211(p):
    x = p[0]
    y = p[1]

    return np.array(ddF(x,y)) + (1/t)*np.linalg.inv(np.dot(np.array(ddG(x,y)), np.array(ddG(x,y))))

print(f"Question 21.1: Inequality Constrained Minimization")
print(f"Optimizing via Newton's Method, starting at (-1,0), with {t} slack.")

x = newtonL2Ineq(f211, constraint211, g211, h211, np.array([-10., -10]))
print(f"Solved! Optimal point: ({x[0]}, {x[1]}).")
print(f"Optimal Function Value: {f211(x)}.")
print(f"Constraint Value: {constraint211(x)}.")

## Problem 21.3: Point internal of dodecahedron closest to external point

t = 10000.
x,y,z = symbols('x y z')
f0 = sqrt((x - 2.2)**2 + (y-1.5)**2 + (z-3.4)**2)
phi = (math.sqrt(5) + 1.)/2.

gs = []
gs.append(-x - phi*y - phi**2)
gs.append( x + phi*y - phi**2)
gs.append(-x + phi*y - phi**2)
gs.append( x - phi*y - phi**2)
gs.append(-y - phi*z - phi**2)
gs.append( y + phi*z - phi**2)
gs.append(-y + phi*z - phi**2)
gs.append( y - phi*z - phi**2)
gs.append(-z - phi*x - phi**2)
gs.append( z + phi*x - phi**2)
gs.append(-z + phi*x - phi**2)
gs.append( z - phi*x - phi**2)

GS = []
for i in range(0,12):
    GS.append(lambdify([x,y,z], gs[i]))

for i in range(0,12):
    f0 = f0 - (1/t)*log(-gs[i])

dF = [diff(f0,x), diff(f0,y), diff(f0,z)]
ddF = [[diff(dF[0],x), diff(dF[0],y), diff(dF[0],z)],
       [diff(dF[1],x), diff(dF[1],y), diff(dF[1],z)],
       [diff(dF[2],x), diff(dF[2],y), diff(dF[2],z)]]

f = lambdify([x, y, z], f0)
df = lambdify([x,y,z], dF)
ddf = lambdify([x,y,z], ddF)

def f213(p):
    x = p[0]
    y = p[1]
    z = p[2]

    return np.array(f(x,y,z))

def g213(p):
    x = p[0]
    y = p[1]
    z = p[2]

    return np.array(df(x,y,z))

def h213(p):
    x = p[0]
    y = p[1]
    z = p[2]

    return np.array(ddf(x,y,z))

def c213(p):
    return -1.

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


x0 = np.zeros((3,))
print(f"Question 21.3: Optimal point within Dodecahedron")
print(f"Optimizing via Newton's Method, starting at {x0[0], x0[1], x0[2]}, with {t} slack.")

x = newtonL2VIneq(f213, GS, g213, h213, x0)
print(f"Solved! Optimal point: ({x[0]}, {x[1]}, {x[2]}).")
print(f"Optimal Function Value: {f213(x)}.")