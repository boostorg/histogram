from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd())

#[ getting_started_listing_04
import histogram as hg

# make 1-d histogram with 5 logarithmic bins from 1e0 to 1e5
h = hg.histogram(hg.axis.regular_log(5, 1e0, 1e5, "x"))

# fill histogram with numbers
for x in (2e0, 2e1, 2e2, 2e3, 2e4):
    h(x, weight=2)

# iterate over bins and access bin counter
for idx, (lower, upper) in enumerate(h.axis(0)):
    print("bin {0} x in [{1}, {2}): {3} +/- {4}".format(
          idx, lower, upper, h.at(idx).value, h.at(idx).variance ** 0.5))

#]
