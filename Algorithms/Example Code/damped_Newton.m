function [x] = damped_Newton(x)
  f = @(x) sqrt(x^2 + 1);
  df = @(x) (x / f(x));
  ddf = @(x) (f(x))^(-3);
  while (norm(df(x)) > 1.e-14)
    d = -ddf(x) \ df(x);
    while (norm(df(x + d)) >= norm(df(x)))
      d *= 0.5;
%      if (abs(d) < 1.e-14) break; end
    end
%    if (abs(d) < 1.e-14) break; end
    x = x + d
end end
