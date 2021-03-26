using LinearAlgebra;  global R2 = zeros(8, 8);  global t = 1.;
function r2(X)  for m in 1 : 7  for n in (m + 1) : 8
  R2[m, n] = (X[m] - X[n])^2 + (X[m + 8] - X[n + 8])^2;  end;  end;  end
function g(X)  G = true;  for n in 1 : 8
  if (X[n]^2 + X[n + 8]^2 >= 1.)  G = false;  end;  end;  return G;  end
function f(X, t)  r2(X);  F = 0.;        # array X = [x1 x2 ... x8 y1 y2 ... y8]
  for m in 1 : 7  for n in (m + 1) : 8  F += 1. / R2[m, n]  end;  end
  for n in 1 : 8  F -= log(1. - X[n]^2 - X[n + 8]^2) / t;  end;  return F;  end
function df(X, t)  r2(X);  G = zeros(16);  for n in 1 : 8
    G[[n, n + 8]] = 2. * [X[n], X[n + 8]] / (t * (1. - X[n]^2 - X[n + 8]^2));
    for m in 1 : 8  if (m != n)  M = min(m, n);  N = max(m, n);
      G[[n, n + 8]] += 2. * [X[m] - X[n], X[m + 8] - X[n + 8]] / R2[M, N]^2;
  end;  end;  end; return G;  end

function BFGS(f, df, X)  F, dF, C = f(X, t), df(X, t), Matrix(1.0 * I, 16, 16);
  while (norm(dF) > 1.e-6 * sqrt(t))
    D = -C * dF;  while ((~g(X + D)) || (f(X + D, t) > F))  D *= 0.9;  end
    X += D;  F = f(X, t);  new_dF = df(X, t);  G = new_dF - dF;  dF = new_dF;
    rho = 1. / (G' * D);  mu = rho * (1. + rho * G' * (C * G));
    DCG = D * (C * G)';  C = C - rho * (DCG + DCG') + mu * (D * D');
  end;  return X;  end

global X = 2. * ones(16);  while (~g(X)) global X = 2. * rand(16) .- 1.;  end
while (t < 1.e+9)  global X = BFGS(f, df, X);  global t *= 2.;  end
for n in 1 : 8  println(X[n], " ", X[n + 8]);  end;  println(f(X, Inf));
