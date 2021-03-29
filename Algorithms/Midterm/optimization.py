import numpy as np
import math
import sys

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

        if np.linalg.norm(DF) < 1e-8:
            break

        DDF = ddF(x)
        if np.isnan(DDF).any() or (np.linalg.cond(DDF) >= 100.):
            print("Bad Hessian")
            break
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

