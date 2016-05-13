Tutorial
========

Example 1: 1d-histogram in C++
------------------------------

How to make a 1d-histogram in C++ and to fill it:

.. code-block:: cpp

    #include <boost/histogram/histogram.hpp>
    #include <boost/histogram/axis.hpp>
    #include <iostream>
    #include <cmath>

    int main(int, char**) {
        namespace bh = boost::histogram;

        /* Create a 1-d histogram with an axis that has 10 bins
         * of equal width, covering the real line in the interval
         * [-1.0, 2.0), label it 'x'.
         * Several other binning strategies are supported, see
         * documentation of axis_types.
         */
        bh::histogram h(bh::regular_axis(10, -1.0, 2.0, "x"));

        /* Fill histogram with a few entries. Values outside of
         * axis are placed in the overflow and underflow bins.
         * Normally you would loop over a source of values.
         */
        h.fill(-1.5); // put in underflow bin
        h.fill(-0.5);
        h.fill(1.1);
        h.fill(-1.0); // included, interval is semi-open
        h.fill(0.3);
        h.fill(1.7);
        h.fill(2.0);  // put in overflow bin, interval is semi-open
        h.fill(20.0); // put in overflow bin

        /* Fill histogram with a weighted count. This increases the
         * bin counter not by one, but by the specified weight.
         *
         * This call transparently causes histogram to change it memory
         * layout to store counts as doubles instead of integers. The
         * layout for weighted counts requires up to 16x more memory
         * and will cause inaccuracies of the type a + 1 == a if a is
         * sufficiently large.
         *
         * Use wfill(...) if you have to, else prefer fill(...).
         */
        h.wfill(0.1, 5.0);

        /* Print a table representation of the histogram showing the bin
         * value and a estimate of the standard deviation. Overflow and
         * Underflow bins are accessed naturally as the bins -1 and 10.
         */
        for (int i = -1; i <= h.bins(0); ++i) {
            const bh::regular_axis& a = h.axis<bh::regular_axis>(0);
            std::cout << "bin " << i
                      << " x in [" << a[i] << ", " << a[i+1] << "): "
                      << h.value(i) << " +/- " << std::sqrt(h.variance(i))
                      << std::endl;
        }
    }

The program output is:

.. code-block:: none

    bin -1 x in [-inf, -1): 1 +/- 1
    bin 0 x in [-1, -0.7): 1 +/- 1
    bin 1 x in [-0.7, -0.4): 1 +/- 1
    bin 2 x in [-0.4, -0.1): 0 +/- 0
    bin 3 x in [-0.1, 0.2): 5 +/- 5
    bin 4 x in [0.2, 0.5): 1 +/- 1
    bin 5 x in [0.5, 0.8): 0 +/- 0
    bin 6 x in [0.8, 1.1): 0 +/- 0
    bin 7 x in [1.1, 1.4): 1 +/- 1
    bin 8 x in [1.4, 1.7): 0 +/- 0
    bin 9 x in [1.7, 2): 1 +/- 1
    bin 10 x in [2, inf): 2 +/- 1.41421


Example 2: 2d-histogram in Python
---------------------------------

How to make a 2d-histogram in Python and to fill it using a Numpy array:

.. code-block:: python

    import histogram as bh
    import numpy as np

    # create a 2d-histogram without underflow and overflow bins
    # for polar coordinates, using a specialized polar_axis for
    # the binning of the angle 'phi'
    #
    # radial axis with label 'radius' has 10 bins from 0.0 to 5.0
    # polar axis with label 'phi' has 4 bins and a phase of 0.0
    h = bh.histogram(bh.regular_axis(10, 0.0, 5.0, "radius",
                                     uoflow=False),
                     bh.polar_axis(4, 0.0, "phi"))

    # fill histogram with random values, using a two-dimensional
    # normal distribution in cartesian coordinates as a source
    x = np.random.randn(1000)             # generate x
    y = np.random.randn(1000)             # generate y
    rphi = np.empty((1000, 2))
    rphi[:, 0] = (x ** 2 + y ** 2) ** 0.5 # compute radius
    rphi[:, 1] = np.arctan2(y, x)         # compute phi
    h.fill(rphi)

    # access counts as a numpy array (no data is copied)
    count_matrix = np.asarray(h)

    print count_matrix

The program output are the counts per bin as a 2d-array:

.. code-block:: python

    [[37 26 33 37]
     [60 69 76 62]
     [48 80 80 77]
     [38 49 45 49]
     [22 24 20 23]
     [ 7  9  9  8]
     [ 3  2  3  3]
     [ 0  0  0  0]
     [ 0  1  0  0]
     [ 0  0  0  0]]
