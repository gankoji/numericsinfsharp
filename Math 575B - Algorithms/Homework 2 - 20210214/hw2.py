from sympy import *
from sympy.vector import gradient
from sympy.vector.coordsysrect import CoordSys3D
import numpy as np
from sympy.utilities.lambdify import lambdify
import math

def f181(inarray):
    x = inarray[0]
    y = inarray[1]

    a = math.exp(8.0*x-13.0*y+21)
    b = math.exp(21.0*y-13.0*x-34)
    c = 0.0001*math.exp(x+y)

    return a + b + c

def gradF(inarray):
    x = inarray[0]
    y = inarray[1]

    a = math.exp(8.0*x-13.0*y+21)
    b = math.exp(21.0*y-13.0*x-34)
    c = 0.0001*math.exp(x+y)

    return np.array([8.0*a-13.0*b+c, -13.0*a+21.0*b+c])

def q_18_1():
    dt = 0.00005

    x = np.array([-1.1, 1.1])
    old_F = 0.
    F = 60000.
    F_x = gradF(x)

    print(f"\n\nMinimizing by Gradient Descent. Starting at ({x[0]}, {x[1]}).")
    while abs(F - old_F) > 1e-6:
        old_x = x
        old_F, F_x, = F, gradF(x)
        x[0], x[1], dt = x[0] - dt*F_x[0], x[1] - dt*F_x[1], 1.1*dt
        F = f181(x)

        if (F > old_F):
            x, F, old_F, dt = old_x, old_F, old_F + 10., 0.5*dt

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

print(f183(np.zeros((99,))))
print(f183(np.ones((99,))))
print(f183(-np.ones((99,))))
