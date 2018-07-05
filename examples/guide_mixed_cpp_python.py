from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd())

#[ guide_mixed_cpp_python_part_py
import histogram as bh
import cpp_filler

h = bh.histogram(bh.axis.regular(4, 0, 1),
                 bh.axis.integer(0, 4))

cpp_filler.process(h)  # histogram is filled with input values in C++

for iy, y in enumerate(h.axis(1)):
    for ix, x in enumerate(h.axis(0)):
        print(h.at(ix, iy).value, end=' ')
    print()

# prints:
# 1.0 0.0 0.0 0.0
# 0.0 1.0 0.0 0.0
# 0.0 0.0 1.0 0.0
# 0.0 0.0 0.0 1.0

#]
