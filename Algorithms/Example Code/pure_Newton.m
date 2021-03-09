function [x] = pure_Newton(x)
  f = @(x) sqrt(x^2 + 1);
  df = @(x) (x / f(x));
  ddf = @(x) (f(x))^(-3);
  while (norm(df(x)) > 1.e-14)
    x = x - ddf(x) \ df(x)
end end
