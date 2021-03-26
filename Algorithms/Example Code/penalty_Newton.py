from numpy import array, sqrt; from numpy.linalg import solve
def df(X, t):
  return 2. * array([X[0] - 1., X[1] - 1., X[2] - 5.]) \
    + 4. * t * (X[0]**2 - X[1]**2 - 2. * X[2]) * array([X[0], -X[1], -1.])
def ddf(X, t):
  return array([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]) \
    + 4. * t * array([[3. * X[0]**2 - X[1]**2 - 2. * X[2], -2. * X[0] * X[1], \
    -2. * X[0]], [-2. * X[0] * X[1], 3. * X[1]**2 - X[0]**2 + 2. * X[2], \
     2. * X[1]], [-2. * X[0], 2. * X[1], 2.]])

X, t = array([3., 0., 0.]), 1.
while (t < 2.e+11):
  while (max(abs(df(X,t))) > 1.e-8 * sqrt(t)):
    X -= solve(ddf(X, t), df(X, t))
  t *= 10.
print(X)
