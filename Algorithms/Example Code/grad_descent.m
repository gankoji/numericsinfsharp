function [x] = grad_descent(f, df, x0, tolerance, factor, alpha, beta)  x = x0;
  while (norm(dF = df(x)) > tolerance)  d = -factor * dF;
    while (f(x + d) >= f(x) + alpha * dF' * d)  d *= beta;  end
    x += d;  end;  end
