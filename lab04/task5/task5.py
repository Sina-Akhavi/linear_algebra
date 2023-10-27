import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np

I = imageio.imread('nasir-al-mulk.jpg')

alpha = np.linspace(0, np.pi, 20)

for single_alpha in alpha:
    s = np.array([np.abs(np.sin(single_alpha)), np.abs(np.sin(single_alpha + np.pi / 4)), np.abs(np.sin(single_alpha + np.pi / 2))])
    J = I * s

    J = np.uint8(J)

    plt.imshow(J)
    plt.draw()
    plt.pause(.1)

print('finished!!!')
plt.show()
