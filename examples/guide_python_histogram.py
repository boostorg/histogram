from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd())

#[ guide_python_histogram
import histogram as hg

# make 1-d histogram with 5 logarithmic bins from 1e0 to 1e5
h = hg.histogram(hg.axis.regular_log(5, 1e0, 1e5, "x"))

# fill histogram with numbers
for x in (2e0, 2e1, 2e2, 2e3, 2e4):
    h(x, weight=4) # increment bin counter by 4

# iterate over bins and access bin counter
for idx, (lower, upper) in enumerate(h.axis(0)):
    print("bin {0} x in [{1}, {2}): {3} +/- {4}".format(
        idx, lower, upper, h.at(idx).value, h.at(idx).variance ** 0.5))

# under- and overflow bins are accessed like in C++
lo, up = h.axis(0)[-1]
print("underflow [{0}, {1}): {2} +/- {3}".format(lo, up, h.at(-1).value, h.at(-1).variance))
lo, up = h.axis(0)[5]
print("overflow  [{0}, {1}): {2} +/- {3}".format(lo, up, h.at(5).value, h.at(5).variance))

# prints:
# bin 0 x in [1.0, 10.0): 4.0 +/- 4.0
# bin 1 x in [10.0, 100.0): 4.0 +/- 4.0
# bin 2 x in [100.0, 1000.0): 4.0 +/- 4.0
# bin 3 x in [1000.0, 10000.0): 4.0 +/- 4.0
# bin 4 x in [10000.0, 100000.0): 4.0 +/- 4.0
# underflow [0.0, 1.0): 0.0 +/- 0.0
# overflow  [100000.0, inf): 0.0 +/- 0.0

#]
