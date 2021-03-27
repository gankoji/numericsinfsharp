## Problem 23.2: Write an RNG that produces numbers distributed according to
## f(x) = x^2, 0<x<1, sqrt(2-x), 1<x<2, 0 o.w.

## von Neumann's rejection method applies here.
from sympy import *
from sympy.functions.elementary.piecewise import Piecewise
import random
import numpy as np
import matplotlib.pyplot as plt
# This is actually a handy little tool I made, keeping it!
def makeBins(binSize, start, end):
    bins = [start]
    while (start <= end):
        start += binSize
        bins.append(start)
    return bins

def problem1():
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

    bins = makeBins(0.025, 0, 2.0)
    plt.hist(samples, bins)
    plt.savefig('Problem_23_2.png')

#problem1()

## Problem 24.1: A mouse's random walk on a graph
## Hard part here will be finding a sensible data structure for the graph
## Solving the probability question via MonteCarlo methods should be 
## straightforward once we've established the algorithm for the walk. 

def problem2():
    ## See Stepanov's notes for the graph
    ## Two choices for a general graph: adjacency matrix or adjacency list
    ## Both have to be built manually :/
    adjacencyMatrix = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])
    ## Walk: recursion would be ideal here, but we're in Python XD
    ## So, a while loop it is. While (not terminated), find node and go. Check termination
    ## Cheese? 1. Cat? 0. 
    def findAdjacents(node, adjMat):
        adjList = []
        for i,el in enumerate(adjMat[node,:]):
            if el:
                adjList.append(i)
        return adjList

    def chooseMove(adjacent):
        n = len(adjacent)
        u = int(random.uniform(0,n))
        return adjacent[u]

    def randomWalkOnGraph(start, adjMat):
        terminated = False
        special = [0, 1] # Cheese is at node 0, cat at node 1
        retval = 0 # 0 means death, 1 means cheese, success, and a happy life
        current = start
        while (not terminated):
            adjacents = findAdjacents(current, adjMat)
            nextNode = chooseMove(adjacents)
            if (nextNode in special):
                terminated = True
                if nextNode == 0:
                    return 1
                else:
                    return 0
            current = nextNode

    ## MonteCarlo simulation: for (large N) runs, store results. Sum, average, boom probability. 
    outputs = []
    nruns = int(1e5)
    for i in range(0, nruns):
        outputs.append(randomWalkOnGraph(0, adjacencyMatrix))

    happinessChance = sum(outputs)/float(nruns)
    print(f"Mouse survives with {happinessChance} probability.")

#problem2()

## Problem 24.5 Consider a diffusing particle inside the square 0<x,y<1. The
## particle starts at x(0) =1/3,y(0) =1/6. Consider the moment and position when and
## where the particle hits the boundary of thesquare for the first time. For each
## side of the square find the probability that it is the one being hit.

## Problem 25.1: Using the Metropolis algorithm, simulate a Markov chain with
## P(x) = exp(-x), x>=0 density and transition function T(x'|x) =
## exp(-(x'-x)^2/2s^2)/sqrt(2\pi)s (with appropriate acceptance probability
## A(x'|x)). Find s that minimizes K_xx(1) = E(x_m-1)(x_{m-1}-1), where x_m is the
## state of the Markov chain at time m.
