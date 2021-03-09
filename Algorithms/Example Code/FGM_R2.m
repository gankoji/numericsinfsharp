function [DATA] = FGM_R2(x, old_x)
  f = @(x) ((x(1) - 1)^2 + 100 * (x(2) - (x(1))^2)^2);
  df = @(x) (2 * [x(1) - 1; 0] + 200 * (x(2) - (x(1))^2) * [-2 * x(1); 1]);
  mu = 0.97; gamma = 0.0012; counter = 0; DATA = zeros(1, 4);
  while (norm(x - old_x) > 1.e-14)
    tmp = x;
    x = x + mu * (x - old_x);
    x = x - gamma * df(x);
    old_x = tmp; counter++;
    DATA(counter, :) = [x(1) x(2) f(x) counter];
end end
