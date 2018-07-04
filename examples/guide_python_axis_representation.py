from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd())

#[ guide_python_axis_representation
import histogram as hg
import numpy as np

ax = hg.axis.regular(5, 0, 1)
xedge1 = np.array(ax) # this is equivalent to...
xedge2 = []
for idx, (lower, upper) in enumerate(ax):
    xedge2.append(lower)
    if idx == len(ax)-1:
        xedge2.append(upper)

print(xedge1)
print(xedge2)

# prints:
# [ 0.   0.2  0.4  0.6  0.8  1. ]
# [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# sequences constructed from an axis use its iterator, the result differs
xedge3 = list(ax)

print(xedge3)

# prints:
# [(0., 0.2), (0.2, 0.4), (0.4, 0.6), (0.8, 1.0)]

#]
