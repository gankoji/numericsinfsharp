import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

# Problem 1.2a
def f0(x):
    return -5.0*x[0] - 3.0*x[1]

c1 = LinearConstraint(np.array([[2., 1.],[1., -1.],[1., 0.]]), np.array([-np.inf, -np.inf, 0]).transpose(), np.array([3, 0, np.inf]).transpose())

res = minimize(f0, np.array([0., 0.]), constraints=c1)

print("Problem 1.2a")
print(f"Optimal value: {res.fun}")
print(f"Optimal point: {res.x}")

# Problem 1.2b
def f1(x):
    return 4.*x[0] -2.*x[1]
    
c = LinearConstraint(np.array([[1., 2.], [1., 0.], [0., 1.]]), np.array([2., 0., 0.]), np.array([np.inf, np.inf, np.inf]))
res = minimize(f1, np.array([0., 0.]), constraints=c)

print("\nProblem 1.2b")
print(f"Optimal value: {res.fun}")
print(f"Optimal point: {res.x}")

# Problem 1.2c
def f2(x):
    return x[3]

A = np.array([[1., 1., 1., -1.],
              [1., -1., 1., -1.],
              [-2., 1., 1., 0.],
              [0., 1., 1., 0.],
              [1., 0., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])

lb = np.array([-np.inf, -np.inf, 4., 6., 0., 0., 0.])
ub = np.array([0., 0., 4., np.inf, np.inf, np.inf, np.inf])
    
c = LinearConstraint(A, lb, ub)
res = minimize(f2, np.array([0., 0., 0., 0.]), constraints=c)

print("\nProblem 1.2c")
print(f"Optimal value: {res.fun}")
print(f"Optimal point: {res.x}")