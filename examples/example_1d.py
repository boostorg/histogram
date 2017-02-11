import histogram as hg
import numpy as np
import matplotlib.pyplot as plt

h = hg.histogram(hg.regular_axis(10, -3, 3))
h.increment(np.random.randn(1000))

bins = h.axis(0).bins

x = np.array(h.axis(0)) # axis instances behave like sequences
y = np.asarray(h)       # creates a view (no copy involved)
y = y[:bins]            # cut off underflow/overflow bins
y = np.append(y, [0])   # append a zero because matplotlib's plot(...) is weird

plt.plot(x, y, drawstyle="steps-post")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
