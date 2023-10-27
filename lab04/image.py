import imageio.v2 as imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

I = imageio.imread('nasir-al-mulk-gray.jpg')
# I_flipped = np.flip(I, axis=0)

plt.imshow(I, cmap='gray')
plt.show()

print('I=\n', I)
print('I.dtype=\n', I.dtype)
print('I.shape=\n', I.shape)

plt.imshow(I.T, cmap='gray')
plt.show()

plt.imshow(I[100:400, 300:600], cmap='gray')
plt.show()
