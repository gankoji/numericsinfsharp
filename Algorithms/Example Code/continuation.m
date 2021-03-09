% minimize (x^2 + y^2) subject to (exp(x) <= y)
% logarithmic barrier, gradual increase of t
function [X] = continuation(X)
  g = @(X) (exp(X(1)) - X(2));
  f = @(X, t) ((X(1))^2 + (X(2))^2 - log(-g(X)) / t);
  t = 1.;
  while (t < 1.e+12)
    while (1 > 0)
      F = f(X, t); G = g(X); ddF = zeros(2); x = X(1); y = X(2);
      dF = 2 * [x; y] + [-exp(x); 1] / (t * G);
      if (norm(dF) < 1.e-7 * sqrt(t)) break; end
      ddF(1, 1) = 2. - exp(x) / (t * G) + exp(2. * x) / (t * G^2);
      ddF(1, 2) = -exp(x) / (t * G^2); ddF(2, 1) = ddF(1, 2);
      ddF(2, 2) = 2. + 1. / (t * G^2);
      d = -ddF \ dF;
% backtracking line search
% want function to go lower, while being in feasible set
      while ((f(X + d, t) >= F) || (g(X + d) >= 0.))
        d *= 0.5;
      end
      X = X + d;
    end
    t *= 2.;
  end
