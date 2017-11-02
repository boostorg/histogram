import histogram as hg
import numpy as np

h = hg.histogram(hg.axis.regular(10, -3, 3, "x"))
h.fill(np.random.randn(1000))

x = np.array(h.axis(0)) # axis instances behave like sequences
y = np.asarray(h)       # creates a view (no copy involved)
y = y[:len(h.axis(0))]  # cut off underflow/overflow bins; y[:-2] also works
y = np.append(y, [0])   # extra zero needed by matplotlib's plot(...) function

try:
    import matplotlib.pyplot as plt
    plt.plot(x, y, drawstyle="steps-post")
    plt.xlabel(h.axis(0).label)
    plt.ylabel("counts")
    plt.savefig("example_1d_python.png")
except ImportError:
    pass