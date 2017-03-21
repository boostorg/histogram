# Histogram

**Fast multi-dimensional histogram with convenient interface for C++11 and Python**

[![Build Status](https://travis-ci.org/HDembinski/histogram.svg?branch=develop)](https://travis-ci.org/HDembinski/histogram?branch=develop) [![Coverage Status](https://coveralls.io/repos/github/HDembinski/histogram/badge.svg?branch=develop)](https://coveralls.io/github/HDembinski/histogram?branch=develop)

This `C++11` library implements two easy-to-use powerful n-dimensional [histogram](https://en.wikipedia.org/wiki/Histogram) classes, using a policy-based design, optimized for extensibility, convenience and highest performance. If the standard configuration is used, bin counts *cannot overflow* or *loose precision*.

Two histogram implementations in C++ are included. `static_histogram` uses compile-time information to provide maximum performance, at the cost of potentially larger executables and reduced runtime flexibility. `dynamic_histogram` makes the opposite trade-off. Python bindings for the latter are included, implemented with `boost.python`.

The histograms have value semantics. Move operations and trips over the language boundary from C++ to Python are cheap. Histograms can be streamed from/to files and pickled in Python. [Numpy](http://www.numpy.org) is supported to speed up operations in Python: histograms can be filled with Numpy arrays at high speed (faster than numpy's own histogram functions) and are convertible into Numpy arrays without copying data.

My goal is to submit this project to [Boost](http://www.boost.org), that's why it uses the Boost directory structure and namespace. The code is released under the [Boost Software License](http://www.boost.org/LICENSE_1_0.txt).

Check out the [full documentation](https://htmlpreview.github.io/?https://raw.githubusercontent.com/HDembinski/histogram/html/doc/html/index.html). Highlights are given below.

## Features

* N-dimensional histogram
* Intuitive and convenient interface
* Value semantics with efficient move operations
* Support for various binning schemes (user-extensible)
* Optional underflow/overflow bins for each dimension
* Bin counts cannot overflow or loose precision (*)
* Support for weighted input
* Statistical variance can be queried for each bin
* High performance (cache-friendly design, tuned code, use of compile-time information to avoid conversions and to unroll loops)
* Space-efficient use of memory, memory dynamically grows as needed
* Serialization support using `boost.serialization`
* Histograms cannot overflow (counts are only limited by available memory)
* Language support: C++11, Python (2.x and 3.x)
* Numpy support

(*) In the standard configuration and if you don't use weighted input.

## Dependencies

* [Boost](http://www.boost.org)
* Optional:
* [CMake](https://cmake.org)
  [Python](http://www.python.org)
  [Numpy](http://www.numpy.org)

## Build instructions

The library can be build with `b2` within the boost directory structure, but if you are not a boost developer, use CMake instead.

```sh
git clone https://github.com/HDembinski/histogram.git
mkdir build && cd build
cmake ../histogram/build
make # or 'make install'
```

To run the tests, do `make test`.

## Code examples

For the full version of the following examples with explanations, see
[Tutorial](https://htmlpreview.github.io/?https://raw.githubusercontent.com/HDembinski/histogram/html/doc/html/tutorial.html).

Example 1: Fill a 1d-histogram in C++

```cpp
    #include <boost/histogram/histogram.hpp> // proposed for inclusion in Boost
    #include <iostream>
    #include <cmath>

    int main(int, char**) {
        namespace bh = boost::histogram;

        // create 1d-histogram with 10 equidistant bins from -1.0 to 2.0,
        // with axis of histogram labeled as "x"
        auto h = bh::make_static_histogram(bh::regular_axis<>(10, -1.0, 2.0, "x"));

        // fill histogram with data
        h.fill(-1.5); // put in underflow bin
        h.fill(-1.0); // included in first bin, bin interval is semi-open
        h.fill(-0.5);
        h.fill(1.1);
        h.fill(0.3);
        h.fill(1.7);
        h.fill(2.0);  // put in overflow bin, bin interval is semi-open
        h.fill(20.0); // put in overflow bin
        h.wfill(0.1, 5.0); // fill with a weighted entry, weight is 5.0

        // access histogram counts, loop includes under- and overflow bin
        for (const auto& b : h.axis<0>()) {
            std::cout << "bin " << b.idx
                      << " x in [" << b.left << ", " << b.right << "): "
                      << h.value(b.idx) << " +/- " << std::sqrt(h.variance(b.idx))
                      << std::endl;
        }

        /* program output:

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
        */
    }
```

Example 2: Fill a 2d-histogram in Python with data in Numpy arrays

```python

    import histogram as bh
    import numpy as np

    # create 2d-histogram over polar coordinates, with
    # 10 equidistant bins in radius from 0 to 5 and
    # 4 equidistant bins in polar angle
    h = bh.histogram(bh.regular_axis(10, 0.0, 5.0, "radius",
                                     uoflow=False),
                     bh.circular_axis(4, 0.0, 2*np.pi, "phi"))

    # generate some numpy arrays with data to fill into histogram,
    # in this case normal distributed random numbers in x and y,
    # converted into polar coordinates
    x = np.random.randn(1000)             # generate x
    y = np.random.randn(1000)             # generate y
    rphi = np.empty((1000, 2))
    rphi[:, 0] = (x ** 2 + y ** 2) ** 0.5 # compute radius
    rphi[:, 1] = np.arctan2(y, x)         # compute phi

    # fill histogram with numpy array
    h.fill(rphi)

    # access histogram counts (no copy)
    count_matrix = np.asarray(h)

    print count_matrix

    # program output:
    #
    # [[37 26 33 37]
    #  [60 69 76 62]
    #  [48 80 80 77]
    #  [38 49 45 49]
    #  [22 24 20 23]
    #  [ 7  9  9  8]
    #  [ 3  2  3  3]
    #  [ 0  0  0  0]
    #  [ 0  1  0  0]
    #  [ 0  0  0  0]]
```

## Benchmarks

Thanks to modern meta-programming and memory management, this library is not only more flexible and convenient to use, but also very fast. In the plot below, its speed is compared to classes from the [ROOT framework](https://root.cern.ch) and to [Numpy](http://www.numpy.org).

![alt benchmark](https://github.com/hdembinski/master/doc/benchmark.png "")

## Rationale

There is a lack of a widely-used free histogram class. While it is easy to write an 1-dimensional histogram, writing an n-dimensional histogram poses more of a challenge. If you add serialization and Python/Numpy support onto the wish-list, the air becomes thin. The main competitor is the [ROOT framework](https://root.cern.ch). This histogram class is designed to be more convenient to use, and as fast or faster than the equivalent ROOT histograms. It comes without heavy baggage, instead it has a clean and modern C++ design which follows the advice given in popular C++ books, like those of [Meyers](http://www.aristeia.com/books.html) and [Sutter and Alexandrescu](http://www.gotw.ca/publications/c++cs.htm).

## Design choices

I designed the histogram based on a decade of experience collected in working with Big Data, more precisely in the field of particle physics and astroparticle physics. I follow these principles:

* "Do one thing and do it well", Doug McIlroy
* The [Zen of Python](https://www.python.org/dev/peps/pep-0020) (also applies to other languages)

### Interface convenience

The histogram has the same consistent interface whatever the dimension. Like `std::vector` it *just works*, users are not forced to make an *a priori* choice among several histogram classes and options everytime they encounter a new data set.

### Language transparency

Python is a great language for data analysis, so the histogram has Python bindings. Data analysis in Python is Numpy-based, so Numpy is supported as well. The histogram can be used as an interface between a complex simulation or data-storage system written in C++ and data-analysis/plotting in Python: define the histogram in Python, let it be filled on the C++ side, and then get it back for further data analysis or plotting.

### Specialized binning strategies

The histogram supports about half a dozent different binning strategies, conveniently encapsulated in axis objects. There is the standard sorting of real-valued data into bins of equal or varying width, but also special support for binning of angles or integer values.

Extra bins that count events with fall outside of the axis range are added by default. This useful feature is activated by default, but can be turned off individually for each axis to conserve memory. The extra bins do not disturb normal counting. On an axis with n-bins, the first bin has the index `0`, the last bin `n-1`, while the under- and overflow bins are accessible at `-1` and `n`, respectively.

### Performance, cache-friendliness and memory-efficiency

Dense storage in memory is a must both for high performance. In the standard configuration, the histograms use a special class which stores the counts in a continuous memory area which grows automatically as needed to hold the largest counts in the histogram. While `std::vector` grows in *length* as new elements are added, while the count storage grows in *depth*.

This scheme is both fast and memory efficient. It is fast, because random access of counts is fast. It is memory efficient, because for small counts only one byte of memory is used per count. Keeping the memory footprint as small as possible also helps to utilitize the CPU cache efficiently.

The scheme works particularly well in light of the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). On the one hand, as the number of histogram axes increases, the number of bins grows exponentially. On the other hand, having many bins reduces the number of counts per bin, which is the other consequence of the rapid increase in volume in an n-dimensional hyper-cube. The smaller memory footprint per bin somewhat compensates the growth of bin numbers.

While the standard configuration is highly recommended, other configurations can be chosen at compile-time. Counts can also be stored in any containers that supports the STL interface and operator[].

### Support for weighted counts and variance estimates

A histogram categorizes and counts, so the natural choice for the data type of the counts are integers. However, in science, histograms are sometimes filled with weighted events, for example, to make sure that two histograms look the same in one variable, while the distribution of another, correlated variable is the subject of study.

In the standard configuration, histogram can be filled with either weighted or unweighted counts. In the weighted case, the sum of weights is stored in a double. The histogram provides a variance estimate is both cases. In the unweighted case, the estimate is computed from the count itself, using a common estimate derived from Poisson-theory. In the weighted case, the sum of squared weights is stored alongside the sum of weights, and used to compute a variance estimate.

## State of project

The histogram is feature-complete. More than 300 individual tests make sure that the implementation works as expected. Comprehensive documentation is available. To grow further, the project needs test users, code review, and feedback.
