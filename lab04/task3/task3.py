import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

I = imageio.imread('nasir-al-mulk-gray.jpg')

# I_tiled = np.tile(I, 2)
# I_reversed = np.flip(I, axis=0)
# I_reversed_tiled = np.tile(I_reversed, 2)
# I_output = np.vstack((I_tiled, I_reversed_tiled))

I_output = np.vstack((np.tile(I, 2), np.tile(np.flip(I, axis=0), 2)))  # solution

plt.imshow(I_output, cmap='gray')

plt.show()

# a = np.array([[1, 2, 3],
#               [6, 7, 8]])
#
# a = np.tile(a, (2, 2))
#
# print(a)
