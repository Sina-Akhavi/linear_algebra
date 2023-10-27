import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np

G = imageio.imread('nasir-al-mulk-gray.jpg')
I = imageio.imread('nasir-al-mulk.jpg')
print(G.shape)


for alpha in np.linspace(0, 1, 20):
    J = alpha * I + (1 - alpha) * G.reshape(853, 1280, 1)

    J = np.uint8(J)

    plt.imshow(J)
    plt.draw()
    plt.pause(.1)

print('finished!!!')
plt.show()

