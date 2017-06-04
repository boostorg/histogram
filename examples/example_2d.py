import histogram as hg
if not "@HAVE_NUMPY@":
    raise SystemExit
import numpy as np

h = hg.histogram(hg.axis.regular(10, -3, 3, uoflow=False),
                 hg.axis.regular(10, -3, 3, uoflow=False))
xy = np.random.randn(2000).reshape((1000, 2))
xy[:,1] *= 0.5
h.fill(xy)

bins = h.axis(0).bins

x = np.array(h.axis(0)) # axis instances behave like sequences
y = np.array(h.axis(1))
z = np.asarray(h)       # creates a view (no copy involved)

try:
    import matplotlib.pyplot as plt
    plt.pcolor(x, y, z.T)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
except ImportError:
    pass
