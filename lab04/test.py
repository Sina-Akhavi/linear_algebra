import numpy as np

a = np.array([[[1, 2, 3], [4, 4, 4]],
             [[6, 7, 8], [9, 9, 9]],
             [[5, 6, 7], [15, 6, 9]]])

print(a.shape)

b = np.random.rand(2)
# b_ = b.reshape(3)

print('b=\n', b)

print('a + b=\n', a + b)
