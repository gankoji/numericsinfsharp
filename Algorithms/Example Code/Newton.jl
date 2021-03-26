function f(x)  V = sqrt(x[3]^2 + x[4]^2);
  return [x[3], x[4], -x[3] * V, -1. - x[4] * V];  end
function f2(theta)  x = 10. * [0., 0., cos(theta), sin(theta)];
  h = 1.e-4;  while (h > 1.e-16) 
    k1 = h * f(x); k2 = h * f(x + 0.5 * k1); k3 = h * f(x + 0.5 * k2);
    k4 = h * f(x + k3);  y = x + (k1 + 2. * k2 + 2. * k3 + k4) / 6.;
    if (y[2] < 0.) h *= 0.5; else x = y;  end;  end;  return -x[1];  end
delta = 1.e-5;  global theta = 0.8;  while (true)  println(theta);
  F = f2.([theta - delta, theta, theta + delta]);
  d = 0.5 * delta * (F[3] - F[1]) / (F[1] - 2. * F[2] + F[3]);
  if (abs(d) > 1.e-9)  global theta -= d;  else  break;  end;  end
