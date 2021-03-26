function [m] = f(X)  A = [3 1; 0 3; -2 -3];  a = A * X;  e = exp(10 * a);
  m = sum(a .* e) / sum(e);  end
