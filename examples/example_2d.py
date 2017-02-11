import histogram as hg
import numpy as np
import matplotlib.pyplot as plt

h = hg.histogram(hg.regular_axis(10, -3, 3, uoflow=False),
                 hg.regular_axis(10, -3, 3, uoflow=False))
xy = np.random.randn(2000).reshape((1000, 2))
xy[:,1] *= 0.5
h.increment(xy)

bins = h.axis(0).bins

x = np.array(h.axis(0)) # axis instances behave like sequences
y = np.array(h.axis(1))
z = np.asarray(h)       # creates a view (no copy involved)

plt.pcolor(x, y, z.T)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
