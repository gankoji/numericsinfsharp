function Nelder_Mead(f, S)  N = size(S)[1];  F = f.(S);
  lo, hi = argmin(F), argmax(F);  while (F[hi] > F[lo] + 1.e-14)
    tmp = F[hi];  F[hi] = -Inf;  n2hi = argmax(F);  F[hi] = tmp;
    xo = (sum(S) - S[hi]) / (N - 1);                        # low facet's center
    xr = 2. * xo - S[hi];  fr = f(xr);                              # reflection
    if (fr < F[lo])
      xe = 2. * xr - xo;  fe = f(xe);                                # expansion
      if (fe < fr)  S[hi], F[hi] = xe, fe;  else  S[hi], F[hi] = xr, fr;  end
     else
      if (fr < F[n2hi])  S[hi], F[hi] = xr, fr;
       else
        xc = 0.5 * (S[hi] + xo);  fc = f(xc);                      # contraction
        if (fc < F[hi])  S[hi], F[hi] = xc, fc;
         else
          for n in 1 : N  if (n != lo)                                  # shrink
            S[n] = 0.5 * (S[lo] + S[n]);  F[n] = f(S[n]);
          end;  end;  end;  end;  end
    lo, hi = argmin(F), argmax(F);  end;  return S[lo], F[lo];  end

function f(X)  t = X[1];  a = X[2];  if (abs(a) > 1.)  return 1.e+100;  end
  return 4 / (sin(t))^2 + 2 / (sin(2 * t))^2 + 3 / 4 + 4 / (1 + a^2) +
    8 * (1 + a^2) / (1 + 2 * a^2 * cos(2 * t) + a^4) + 1 / (4 * a^2);  end
println(Nelder_Mead(f, [rand(2) for n in 1 : 3]));
