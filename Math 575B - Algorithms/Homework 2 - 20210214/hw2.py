from sympy import *
from sympy.vector import gradient
from sympy.vector.coordsysrect import CoordSys3D
from sympy.utilities.lambdify import lambdify
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

def f181(inarray):
    x = inarray[0]
    y = inarray[1]

    a = math.exp(8.0*x-13.0*y+21)
    b = math.exp(21.0*y-13.0*x-34)
    c = 0.0001*math.exp(x+y)

    return a + b + c

def g181(inarray):
    x = inarray[0]
    y = inarray[1]

    a = math.exp(8.0*x-13.0*y+21)
    b = math.exp(21.0*y-13.0*x-34)
    c = 0.0001*math.exp(x+y)

    return np.array([8.0*a-13.0*b+c, -13.0*a+21.0*b+c])

def q_18_1():
    dt = 0.00005
    x = np.array([-1.1, 1.1])
    F = 60000.

    print(f"\n\nMinimizing by Gradient Descent. Starting at ({x[0]}, {x[1]}).")
    x = gradientDescent(f181, g181, x, F, dt)
    print("\nSolution Found. ")
    print(f"Optimal Solution Point: ({x[0]}, {x[1]})")
    print(f"Objective Function Value: {F}")

q_18_1()

def f(inarray):
    x = inarray[0]
    y = inarray[1]

    a = 3*x*y - 2*y
    b = (inarray[0]**2 + inarray[1]**2 - 1.1)
    c = 1000*b*math.exp(10*b)

    return a + b + c

def gradF(inarray):
    x = inarray[0]
    y = inarray[1]

    a = 3.*x*y - 2.*y
    b = x**2.+y**2.-1.1
    c = math.exp(10.*b)

    return np.array([3.*y + 1000.*(2.*x*c + 20*x*b*c), 3.*x-2. + 1000.*(2.*y*c + 20*y*b*c)])

def q_18_2():
    x = np.array([-0.1, -0.1])
    F = f(x) 
    F_x = gradF(x)
    C = np.eye(x.size)

    print(f"\n\nMinimizing by BFGS. Starting at ({x[0]}, {x[1]}).")
    while np.linalg.norm(F_x) > 1e-6:
        d = np.dot(-C,F_x)
        while(f(x+d) >= F):
            d = 0.0005*d
            if (np.linalg.norm(d) < 1e-4):
                C = np.eye(x.size)
                d = -F_x
            
        x = x+d
        F = f(x)
        
        g = gradF(x) - F_x
        F_x = gradF(x)
        # a = np.dot(g.transpose(),d)
        # if math.fabs(a) < 1e-3:
        #     a = 1

        # rho = 1./a
        # mu = rho*(1.+rho*np.dot(g.transpose(), np.dot(C,g)))
        # C = C - rho*(np.dot(d, np.dot(g.transpose(),C)) + np.dot(C, np.dot(g,d.transpose())) + mu*np.dot(d,d.transpose()))
        A = np.eye(x.size) - np.outer(d, g)/np.inner(d,g)
        B = np.eye(x.size) - np.outer(g, d)/np.inner(g,d)
        D = np.outer(d,d)/np.inner(g,d)
        C = np.dot(A, np.dot(C,B)) + D

        print(C)


    print("\nSolution Found. ")
    print(f"Optimal Solution Point: ({x[0]}, {x[1]})")
    print(f"Objective Function Value: {F}")

#q_18_2()

def f183(inarray):
    x = inarray
    x = np.insert(x, 0, -1., axis=0)
    x = np.insert(x, 100, 1., axis=0)

    sum = 0.0
    for i in range(0,100):
        sum += 0.5*((x[i+1] - x[i])**2) + 0.0625*((1 - x[i]**2)**2)

    return sum

def g183(inarray): 
    x = inarray
    x = np.insert(x, 0, -1., axis=0)
    x = np.insert(x, 100, 1., axis=0)
    g = np.zeros(inarray.shape)
    for i in range(1, 100):
        a = x[i-1]
        b = x[i]
        c = x[i+1]
        g[i-1] = 2*b - c - a - 0.25*(b - b**3)

    # print(inarray)
    # print(f183(inarray))
    # print(g)
    return g

def q_18_3():
    dt = 0.1
    x = 0.1*np.ones((99,))
    F = f183(x) + 100.

    print(f"\n\nMinimizing by Gradient Descent. Starting at ({x}).")
    print(f"Initial gradient: {g183(x)}.")
    x = gradientDescent(f183, g183, x, F, dt)
    print("\nSolution Found. ")
    print(f"Optimal Solution Point: {x}")
    print(f"Objective Function Value: {F}")

#q_18_3()
# print(g183(np.zeros((99,))))
# print(f183(np.ones((99,))))
# print(g183(-np.ones((99,))))
