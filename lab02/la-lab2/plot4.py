import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u = np.array([1,2,3])
v = np.array([2.0, 0, -2])

rng = np.linspace(0,1,20)

for alpha in rng:
    plt.cla()
    w = (1-alpha) * u + alpha * v

    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    ax.set_zlim(-4,4)

    ax.quiver(0,0,0, u[0], u[1], u[2], color='r')
    ax.quiver(0,0,0, v[0], v[1], v[2], color='r')

    ax.quiver(0,0,0, w[0], w[1], w[2], color='b')
    ax.scatter(w[0], w[1], w[2], color='b')
    plt.draw()
    plt.pause(0.5)
    
plt.show()
