import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

## Problem 24.3: Importance Sampling
def problem_24_3():
    # Our first attempt will be simple/naive MC
    N = int(1e5) # number of samples

    data = np.random.normal(0,1,N) # N samples of the normal distribution
    ev = 0.0
    for x in data:
        ev += (x**20.0)/N

    print(f"Naive MC Expected value: {ev}")
    # This gives a (predictably) terrible estimate of a very large number, when the answer should be near 0. 

    # Next up is Importance Sampling
    N = int(1e5) # number of samples

    fx = norm(0,1)
    g = norm(0,1e-1)
    data = np.random.normal(0,1e-1,N) # N samples of the normal distribution
    ev = 0.0
    for x in data:
        scale = fx.pdf(x)/g.pdf(x)
        ev += scale*(x**20.0)/N

    print(f"Importance Sampled Expected value: {ev}")

## Problem 24.6: Markov Chains
def T(x,y):
    if y <= (x+1):
        return 1/(x+2)
    else:
        return 0

def factorial(x):
    if x < 2:
        return 1
    else: 
        return x*factorial(x-1)

econst = math.exp(-1)
def pi0(x):
    return econst*factorial(x)

def MetropolisHastings(x):
    newx = x + np.random.randn()
    newT = T(newx,x)
    oldT = T(x,newx)
    if math.fabs(newT) < 1e-4:
        # accept the new sample
        return newx
    elif math.fabs(oldT) < 1e-4:
        # skip the actual sampling, its expensive
        return x
    else:
        newP = pi0(newx)
        oldP = pi0(x)
        if math.fabs(oldP) < 1e-4:
            # accept the new sample
            return newx

        A = max(1, (newP/oldP)*(oldT/newT))
        u = np.random.rand()

        if u <= A:
            return newx
    return x

def problem_24_6_a():
    nsamples = int(1e6)
    xs = np.zeros((nsamples,))

    for i in range(1,nsamples):
        xs[i] = MetropolisHastings(xs[i-1])

    bins = np.linspace(-2000, 0, 100)
    plt.hist(xs, bins)
    plt.show()
    
def MetropolisHastings2(x):
    newx = np.random.randint(0,x+2)
    newT = T(newx,x)
    oldT = T(x,newx)
    if math.fabs(newT) < 1e-4:
        # accept the new sample
        return newx
    elif math.fabs(oldT) < 1e-4:
        # skip the actual sampling, its expensive
        return x
    else:
        A = max(1, (oldT/newT))
        u = np.random.rand()

        if u <= A:
            return newx
        
    return x

def problem_24_6_b():
    # Took about 5 hours running 5e8 samples
    # Probability was about 1.04e-7
    N = 1e6
    nsamples = int(N)
    successes = np.zeros((nsamples,))

    oldx = 0
    for i in range(0,nsamples):
        for j in range(0,15):
            x = MetropolisHastings2(oldx)
            oldx = x
        
        if x > 8:
            print(x)
        if math.fabs(x-10.) < 1e-1:
            successes[i] = 1.
        else:
            successes[i] = 0.

    probSuccess = np.sum(successes)/float(N)
    print(f"Probability of ending at X=10: {np.sum(successes)}")

## Problem 25.2: 2D Ising Model with Glauber Dynamics
## Basically going to copy the example code from the text, porting to Python.
def slickerNeighbors(S, m,n,i,j):
    # Our standard 'cross' shape for neighbors
    startpairs = [(i-1,j),(i,j),(i+1,j),(i,j-1),(i,j+1)]
    pairs = []
    for p in startpairs:
        x = p[0]
        y = p[1]
        if x < 0:
            x = x + m
        if x >= m:
            x = x - m
        if y < 0:
            y = y + n
        if y >= n:
            y = y - n

            pairs.append((x,y))
    acc = 0.0    
    for p in pairs:
        acc += S[p[0]][p[1]]

    return 2.*(acc - 2.)
                
def IsingSpinRuns(S, nruns, T):
    p = S.shape # The size of S
    m,n = p[0], p[1] # Size of x and y dimensions, respectively
    for k in range(0,nruns):
        # Uniformly sample a location in S (integers!)
        i = np.random.randint(0,m)
        j = np.random.randint(0,n)

        # At the place we've sampled, calculate E based on neighbors
        E = slickerNeighbors(S,m,n,i,j)
        thresh = 1./(1+ math.exp(-2.*E/T))
        if np.random.rand() < thresh:
            S[i][j] = 1
        else:
            S[i][j] = 0

    return S

def problem_25_2():
    ntemps = 5
    nloops = 40
    nsamples = 5000

    Ls = [32, 64, 128, 256]
    Temps = np.linspace(2,3,ntemps)

    for L in Ls:
        for T in Temps:
            S = np.ones((L,L))
            mags = np.zeros((nloops,))
            for j in range(0,nloops):
                S = IsingSpinRuns(S,nsamples,T)
                M = math.fabs(np.sum(S)/(L**2))
                mags[j] = M

            plt.figure()
            plt.plot(mags)
            plt.title(f"Magnetization for T={T}")
            plt.xlabel("Loop iterations")
            plt.ylabel("Average magnetization")
            plt.savefig(f"Mag_{T}_L_{L}.png")

#problem_24_3()
#problem_24_6_a()
problem_24_6_b()
#problem_25_2()

