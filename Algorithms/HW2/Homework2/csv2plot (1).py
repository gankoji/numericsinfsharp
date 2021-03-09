import numpy as np
import matplotlib.pyplot as plt

ls = ["1.000000", "2.000000", "4.000000", "8.000000", "16.000000", "32.000000", "64.000000", "128.000000", "256.000000"]

for l in ls:
    print(f"Lambda: {l}")
    fileName = "Denoised_" + l + ".csv"
    imageName = "Denoised_" + l + ".png"
    image = np.genfromtxt(fileName, delimiter=',')
    plt.imshow(image)
    plt.savefig(imageName)
plt.show()