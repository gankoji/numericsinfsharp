from math import pi, sqrt, cos, sin;  from numpy.random import rand
def f(X):  return 3. * X[3] - 2. * X[1]                   # X = [x1, y1, x2, y2]
r, best_X, best_f, X, counter = sqrt(2.), rand(4), 1.e+100, rand(4), 0
while (True):
  X[0] = 1 + rand();  X[1] = X[0];              # generating x1, y1 with x1 = y1
  while (True):        # generating x2, y2 so that (x1 - x2)^2 + (y1 - y2^2) = 2
    X[2 : 4] = X[0 : 2] + [r * cos(a := 0.25 * pi * (5. + rand())), r * sin(a)]
    if ((X[2] >= X[3]) and (X[2]**2 + X[3]**2 <= 4. * X[3])):  break
  counter += 1                             # updating the best point seen so far
  if (f(X) < best_f):  best_f, best_X = f(X), X;  print(best_X, best_f, counter)
