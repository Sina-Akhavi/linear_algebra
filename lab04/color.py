import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np

I = imageio.imread('nasir-al-mulk.jpg')

print('I.shape=', I.shape)

plt.imshow(I)
plt.show()
