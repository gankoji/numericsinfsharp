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

## Problem 3: Trading cards in cookie boxes
from random import random
from scipy import interpolate
def problem3():
    ## The approach is straightforward
    ## We build the PMF for each N by monte carlo sampling
    ## Having the PMF, we can directly answer part 1, and integrate/sum
    ## it to get the answer for part 2

    ## In true programmer fashion, our cards are the integers 0-99
    def getCard():
        temp = 100*random() # a float between 0.00 and 99.999999...
        return int(temp)

    ## For a single run of N boxes
    ## Get N cards, increment the number of each card found
    ## Check for any zeros in our set. If any, we have failed to
    ## get a full set. Otherwise, we have succeeded. 
    def runTrial(N):
        cards = np.zeros((100,))
        for i in range(0,N):
            card = getCard()
            cards[card] += 1

        return not (np.any(cards == 0))

    ## Finally, build the PMF.
    ## Let's go from 100 to 1000 for now
    ## Obviously, we can't get a full set with less than 100... 
    probs = []
    numTrials = int(1e4)
    ns = np.logspace(2,4,200,dtype=int)
    for n in ns:
        prob = 0.00
        for m in range(0,numTrials):
            prob += float(runTrial(n))

        prob = prob/numTrials
        probs.append(prob)
    pmf = np.array(probs)
    pmf = pmf/np.sum(pmf)
    cmf = np.zeros(pmf.shape)
    for i in range(0,len(cmf)):
        cmf[i] = np.sum(pmf[0:i+1])

    plt.plot(ns,pmf)
    plt.xlabel("k, Number of cookie boxes")
    plt.ylabel("Probability of full set")
    plt.title("Probability Mass Function p(N=k)")
    plt.savefig("p3_pmf.png")

    plt.figure()
    plt.plot(ns,cmf)
    plt.xlabel("k, Number of cookie boxes")
    plt.ylabel("Probability of full set")
    plt.title("Cumulative Distribution Function p(N<=k)")
    plt.savefig("p3_cmf.png")

    ## Finally, let's answer the cumulative probability question
    ## First, build an interpolation object with SciPy's handy utility class
    cmf_interp = interpolate.interp1d(ns, cmf)

    ## Next, get the answer for N=500
    p500 = cmf_interp(500)

    ## Finally, print it out nice and pretty
    print(f"The probability that a full collection is obtained from 500 boxes or less: {p500}.")
    plt.show()


## Problem 4: A sum of IID RVs
def problem4():
    ## Again, a very simple approach. 
    ## Sample X directly, 
    def getX():
        # Generate 48 IID uniform samples
        xs = np.random.uniform(0,1,size=48)
        return np.sum(xs)

    ## and build it's statistics via a monte carlo approach.
    def buildCDF(n):
        xs = np.zeros((n,))
        for i in range(0,n):
            xs[i] = getX()

        xs = np.sort(xs)
        ys = np.linspace(0,1,n)
        return xs, ys

    n = int(1e8)
    xs,ys = buildCDF(n)
    cdf_interp = interpolate.interp1d(xs,ys)
    try:
        p36 = cdf_interp(36)
        print(p36)
    except ValueError:
        print("None of our samples reach 36. Thus, our estimate for P(X>36) is zero.")
        p36 = 0.0
    plt.figure()
    plt.plot(xs,ys)
    plt.axvline(36)
    plt.text(26,0.5,f"P(X>36):{p36}")
    plt.show()
# Each problem is defined as a function above. Here we actually run them!
# (This just lets me easily run one at a time while I'm developing the solution)

# problem1()
# problem2(1.)
# problem2(2.)
# problem2(3.)
# problem3()
problem4()