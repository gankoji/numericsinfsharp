h = @(X) ((X(1) - X(3))^2 + (X(2) - X(4))^2 - 2.);
dh = @(X) 2. * [X(1) - X(3); X(2) - X(4); X(3) - X(1); X(4) - X(2)];
g1 = @(x, y) y - x;  g2 = @(x, y) x^2 + y^2 - 4. * y;
g = @(X) [g1(X(1), X(2)); g1(X(3), X(4)); g2(X(1), X(2)); g2(X(3), X(4))];
f = @(X, t) 3. * X(4) - 2. * X(2) + t * h(X)^2 - log(prod(g(X))) / t;
df = @(X, t) [0.; -2.; 0.; 3.] + 2. * t * h(X) * dh(X) ...
 + ([1.; -1.; 0.; 0.] / g1(X(1), X(2)) + [0.; 0.; 1.; -1.] / g1(X(3), X(4)) ...
 - 2. * [X(1); X(2) - 2.; 0.; 0.] / g2(X(1), X(2)) ...
 - 2. * [0.; 0.; X(3); X(4) - 2.] / g2(X(3), X(4))) / t;

X = [1.2; 1.1; 0.9; 0.8];  t = 1.;  while (t < 1.e+14)
  F = f(X, t);  dF = df(X, t);  C = eye(length(X));
  while (norm(dF) > 1.e-7 * sqrt(t))  D = -C * dF;         # direction of search
    while ((f(X + D, t) >= F) || any(g(X + D) >= 0.))  D *= 0.9;  end
    X = X + D;  F = f(X, t);  new_dF = df(X, t);  G = new_dF - dF;  dF = new_dF;
    rho = 1. / (G' * D);  mu = rho * (1. + rho * (G' * C * G));    # BFGS update
    C = C - rho * (D * (G' * C) + (C * G) * D') + mu * D * D';     # of ddf^(-1)
  end;  t *= 10.;  end              # gradual increase of t, continuation method
