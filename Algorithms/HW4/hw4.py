## Problem 23.2: Write an RNG that produces numbers distributed according to
## f(x) = x^2, 0<x<1, sqrt(2-x), 1<x<2, 0 o.w.

## von Neumann's rejection method applies here.
from sympy import *
from sympy.functions.elementary.piecewise import Piecewise
import random
import numpy as np
import matplotlib.pyplot as plt

x = symbols('x')
f = Piecewise((x**2, ((x >= 0) & (x <= 1))), (sqrt(2-x), ((x>1) & (x<=2))), (0, True))
f = lambdify([x], f)
nsamples = int(1e5)
samples = np.zeros((nsamples,))
for i in range(0,nsamples):
    success = False
    while (not success):
        u1 = random.uniform(0,2)
        u2 = random.uniform(0,1)
        ythresh = f(u1)
        if u2 <= ythresh:
            samples[i] = u1
            success = True

# This is actually a handy little tool I made, keeping it!
def makeBins(binSize, start, end):
    bins = [start]
    while (start <= end):
        start += binSize
        bins.append(start)
    return bins

bins = makeBins(0.025, 0, 2.0)
plt.hist(samples, bins)
plt.savefig('Problem_23_2.png')