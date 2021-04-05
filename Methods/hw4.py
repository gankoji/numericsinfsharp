# Problem 8.6 from the course notes. 

## Calculate p(x), p(y), p(y|x), p(x|y), H(X), H(Y), I(Y;X) from the below table
import numpy as np
import math

pxy = np.array([[ 1/8, 1/16, 1/32, 1/32],
                [1/16,  1/8, 1/32, 1/32],
                [1/16, 1/16, 1/16, 1/16],
                [ 1/4,    0,    0,    0]])

px = np.sum(pxy, axis=1)
py = np.sum(pxy, axis=0)

pygx = np.zeros(pxy.shape)
for i in range(0,4):
    for j in range(0,4):
        pygx[i,j] = pxy[i,j]/px[j]

pxgy = np.zeros(pxy.shape)
for i in range(0,4):
    for j in range(0,4):
        pxgy[i,j] = pxy[i,j]/py[j]

print(pxy)
hxy = 0.

for i in range(0,4):
    for j in range(0,4):
        if pxy[i,j] != 0.:
            hxy += -math.log(pxy[i,j])*pxy[i,j]
print(hxy)