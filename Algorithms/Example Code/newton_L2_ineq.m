function [X] = newton_L2_ineq(X, t)
  g = @(x) ((x(2))ˆ2 - (x(1))ˆ2 * (1. - 2. * x(1) / 3.));
  f = @(x) ((x(1) - 1.)ˆ2 + (x(2) - 1.)ˆ2 - log(-g(x)) / t);

  while (1 > 0)
    F = f(X);
    x = X(1); y = X(2); G = yˆ2 - xˆ2 + 2. * xˆ3 / 3.;
    dF = 2. * [x - 1.; y - 1.] - 2. * [xˆ2 - x; y] / (t * G);

    if (norm(dF) < 1.e-8)
        break; 
    end

    ddF = [0., 0.; 0., 2. + 4. * yˆ2 / (t * Gˆ2) - 2. / (t * G)];
    ddF(1, 2) = 4. * (xˆ2 - x) * y / (t * Gˆ2); ddF(2, 1) = ddF(1, 2);
    ddF(1, 1) = 2. + 4. * (xˆ2 - x)ˆ2 / (t * Gˆ2) - (4. * x - 2.) / (t * G);

    d = -ddF \ dF;

    while ((f(X + d) >= F) || (g(X + d) >= 0.))
        d = d / 2.;
    end

    X = X + d; 
  end