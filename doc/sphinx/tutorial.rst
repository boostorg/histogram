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

        // create histogram
        bh::histogram h(bh::regular_axis(10, -1.0, 2.0, "x"));

        // fill histogram
        h.fill(-1.0);
        h.fill(-0.5);
        h.fill(1.1);

        // fill histogram with weighted count
        h.wfill(0.1, 5.0);

        // print counts and variance estimate to stdout,
        // including underflow and overflow bins
        for (int i = -1; i <= h.bins(0); ++i) {
            std::cout << "bin " << i << " " << h.value(i) 
                      << " +/- " << std::sqrt(h.variance(i))
                      << std::endl;
        }
    }

Example 2: 2d-histogram in Python
---------------------------------

How to make a 2d-histogram in Python and to fill it using a Numpy array:

.. code-block:: python

    import histogram as bh
    import numpy as np

    # create histogram without underflow and overflow bins
    h = bh.histogram(bh.regular_axis(10, 0.0, 5.0, "radius",
                                     uoflow=False),
                     bh.polar_axis(4, 0.0, "phi"))

    # fill histogram
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    rphi = np.empty((1000, 2))
    rphi[:, 0] = (x ** 2 + y ** 2) ** 0.5
    rphi[:, 1] = np.arctan2(y, x)
    h.fill(rphi)

    # access counts as a numpy array (no data is copied)
    count_matrix = np.asarray(h)

    print count_matrix

