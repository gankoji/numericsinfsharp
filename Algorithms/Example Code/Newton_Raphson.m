function [x] = Newton_Raphson(f, x0)  n = length(x0);  delta = 5.e-6;
  x_old = x0; x = x_old; x_old(1) = x_old(1) + 1.;  dx = 0.5 * delta * eye(n);
  while (norm(x - x_old) > 1.e-12)  x_old = x;  dfdx = zeros(n);
    for i = 1 : n
      dfdx(:, i) = (f(x + dx(:, i)) - f(x - dx(:, i))) / delta;  end;
  x = x_old - dfdx \ f(x_old);  end
