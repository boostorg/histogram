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
h.fill(x, y)

# get representations of the bin edges as Numpy arrays, this representation
# differs from `list(h.axis(0))`, because it is optimised for compatibility
# with existing Numpy code, i.e. to replace numpy.histogram
x = np.array(h.axis(0))
y = np.array(h.axis(1))

# creates a view of the counts (no copy involved)
count_matrix = np.asarray(h)

# cut off the under- and overflow bins (no copy involved)
reduced_count_matrix = count_matrix[:-2,:-2]

try:
    # draw the count matrix
    import matplotlib.pyplot as plt
    plt.pcolor(x, y, reduced_count_matrix.T)
    plt.xlabel(h.axis(0).label)
    plt.ylabel(h.axis(1).label)
    plt.savefig("example_2d_python.png")
except ImportError:
    # ok, no matplotlib, then just print it
    print count_matrix
