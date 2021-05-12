import math
import numpy as np

## Unconstrained solver
def gradientDescent (f, gradF, x0, f0, dt0, tol):
    x = x0
    fval = f0
    dt = dt0
    newf = f(x)
    newg = gradF(x)

    while (math.fabs (newf - fval)) > tol:
        fval = newf
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

def newtonIneq(f,gs, dF, ddF, xin, tol):
    x = xin
    t = 1.
    while (t < 1e14):
        #print("Increasing t")
        cur_dF = dF(x, t)
        while (np.linalg.norm(cur_dF) > tol*math.sqrt(t)):
            #print("Stepping")
            d = -np.linalg.solve(ddF(x,t),cur_dF)
            nx = x+d
            gvals = np.array([g(nx[0],nx[1],nx[2],t) for g in gs])
            while(np.any(gvals > 0)):
                #print("Halving step")
                d = d/2.
                nx = x+d
                gvals = np.array([g(nx[0],nx[1],nx[2],t) for g in gs])
            cur_F = f(x,t)
            while (f(x+d,t) >= cur_F):
                # print("Halving step for fun")
                # print(d)
                # print(f(x+d,t))
                # print(cur_F)
                d = d/2.
                if np.linalg.norm(d) <= tol:
                    break
            x = x+d
            cur_dF = dF(x,t)
        t = t*2.
    
    return x


