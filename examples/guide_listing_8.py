import histogram as bh
import numpy as np

h = bh.histogram(bh.axis.integer(0, 9))

# don't do this, it is very slow
for i in range(10):
	h.fill(i)

# do this instead, it is very fast
v = np.arange(10)
h.fill(v) # fills the histogram with each value in the array
