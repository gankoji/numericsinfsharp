function f(x)  V = sqrt(x[3]^2 + x[4]^2);
  return [x[3], x[4], -x[3] * V, -1. - x[4] * V];  end
function f2(theta)  x = 10. * [0., 0., cos(theta), sin(theta)];
  h = 1.e-4;  while (h > 1.e-16) 
    k1 = h * f(x); k2 = h * f(x + 0.5 * k1); k3 = h * f(x + 0.5 * k2);
    k4 = h * f(x + k3);  y = x + (k1 + 2. * k2 + 2. * k3 + k4) / 6.;
    if (y[2] < 0.) h *= 0.5; else x = y;  end;  end;  return -x[1];  end
function golden_search(x0, x1)  a = (sqrt(5.) - 1.) / 2.; b = 1. - a;
  x = [x0, a * x0 + b * x1, x1];  f = f2.(x);
  while (abs(x[1] - x[3]) > 1.e-10)  xm = a * x[2] + b * x[3]; fm = f2(xm);
    if (fm < f[2])  x = [x[2], xm, x[3]]; f = [f[2], fm, f[3]];
     else  x = [xm, x[2], x[1]]; f = [fm, f[2], f[1]];  end
  end;  return [x[2], -f[2]];  end
println("[optimal theta, maximal firing range] = ", golden_search(0., pi / 2.));
