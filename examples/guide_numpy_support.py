from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd())

#[ guide_numpy_support
import histogram as hg
import numpy as np

# create 2d-histogram with two axes with 10 equidistant bins from -3 to 3
h = hg.histogram(hg.axis.regular(10, -3, 3, "x"),
                 hg.axis.regular(10, -3, 3, "y"))

# generate some numpy arrays with data to fill into histogram,
# in this case normal distributed random numbers in x and y
x = np.random.randn(1000)
y = 0.5 * np.random.randn(1000)

# fill histogram with numpy arrays, this is very fast
h(x, y) # call looks the same as if x, y were values

# get representations of the bin edges as Numpy arrays; this representation
# differs from `list(h.axis(0))` as explained in the next example
x = np.array(h.axis(0))
y = np.array(h.axis(1))

# creates a view of the counts (no copy involved)
count_matrix = np.asarray(h)

# cut off the under- and overflow bins to not confuse matplotib (no copy)
reduced_count_matrix = count_matrix[:-2,:-2]

try:
    # draw the count matrix
    import matplotlib.pyplot as plt
    plt.pcolor(x, y, reduced_count_matrix.T)
    plt.xlabel(h.axis(0).label)
    plt.ylabel(h.axis(1).label)
    plt.savefig("example_2d_python.png")
except ImportError:
    # ok, no matplotlib, then just print the full count matrix
    print(count_matrix)

    # output of the print looks something like this, the two right-most rows
    # and two down-most columns represent under-/overflow bins
    # [[ 0  0  0  1  5  0  0  1  0  0  0  0]
    #  [ 0  0  0  1 17 11  6  0  0  0  0  0]
    #  [ 0  0  0  5 31 26  4  1  0  0  0  0]
    #  [ 0  0  3 20 59 62 26  4  0  0  0  0]
    #  [ 0  0  1 26 96 89 16  1  0  0  0  0]
    #  [ 0  0  4 21 86 84 20  1  0  0  0  0]
    #  [ 0  0  1 24 71 50 15  2  0  0  0  0]
    #  [ 0  0  0  6 26 37  7  0  0  0  0  0]
    #  [ 0  0  0  0 11 10  2  0  0  0  0  0]
    #  [ 0  0  0  1  2  3  1  0  0  0  0  0]
    #  [ 0  0  0  0  0  2  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  1  0  0  0  0  0  0]]

#]
