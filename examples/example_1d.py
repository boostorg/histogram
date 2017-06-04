import histogram as hg
if "@HAVE_NUMPY@":
    raise SystemExit
import numpy as np

h = hg.histogram(hg.axis.regular(10, -3, 3))
h.fill(np.random.randn(1000))

bins = h.axis(0).bins

x = np.array(h.axis(0)) # axis instances behave like sequences
y = np.asarray(h)       # creates a view (no copy involved)
y = y[:bins]            # cut off underflow/overflow bins
y = np.append(y, [0])   # append a zero because matplotlib's plot(...) is weird

try:
    import matplotlib.pyplot as plt
    plt.plot(x, y, drawstyle="steps-post")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
except ImportError:
    pass