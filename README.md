# Histogram

**Fast n-dimensional histogram with convenient interface for C++ and Python**

[![Build Status](https://travis-ci.org/HDembinski/histogram.svg?branch=master)](https://travis-ci.org/HDembinski/histogram) [![Coverage Status](https://coveralls.io/repos/github/HDembinski/histogram/badge.svg?branch=master)](https://coveralls.io/github/HDembinski/histogram?branch=master)

This project contains an easy-to-use powerful n-dimensional [histogram](https://en.wikipedia.org/wiki/Histogram) class implemented in `C++03`-compatible code, optimized for convenience and excellent performance under heavy duty. Move semantics are supported using `boost::move`. The histogram has a complete [C++](http://yosefk.com/c++fqa/defective.html) and [Python](http://www.python.org) interface, and can be passed over the language boundary with ease. [Numpy](http://www.numpy.org) is fully supported; histograms can be filled with Numpy arrays at C speeds and are convertible into Numpy arrays without copying data. Histograms can be streamed from/to files and pickled in Python.

My goal is to submit this project to [Boost](http://www.boost.org), that's why it uses the Boost directory structure and namespace. The code is released under the [Boost Software License](http://www.boost.org/LICENSE_1_0.txt).

[Full documentation](https://htmlpreview.github.io/?https://raw.githubusercontent.com/HDembinski/histogram/master/doc/html/index.html) is available, a summary is given below.

## Features

* N-dimensional histogram
* Intuitive and convenient interface
* Support for different binning schemes, including binning of angles
* Optional underflow- and overflow-bins for each dimension
* Support for weighted events, with variance estimates for each bin
* Support for move semantics using `boost::move` in `C++03` and `C++0x`
* High performance through cache-friendly design
* Space-efficient memory storage that dynamically grows as needed
* Serialization support with zero-suppression
* Multi-language support: C++ and Python
* Numpy support

## Dependencies

* [Boost](http://www.boost.org)
* [CMake](https://cmake.org)
* Optional:
  [Python](http://www.python.org)
  [Numpy](http://www.numpy.org)
  [Sphinx](http://www.sphinx-doc.org)

## Build instructions

```sh
git clone https://github.com/HDembinski/histogram.git
mkdir build && cd build
cmake ../histogram/build
make install # (or just 'make' to run the tests)
```

To run the tests, do `make test` or `ctest -V` for more output.

## Code examples

For the full version of the following examples with explanations, see
[Tutorial](https://htmlpreview.github.io/?https://raw.githubusercontent.com/HDembinski/histogram/master/doc/html/tutorial.html).

Example 1: Fill a 1d-histogram in C++ 

```cpp
    #include <boost/histogram/histogram.hpp> // proposed for inclusion in Boost
    #include <boost/histogram/axis.hpp> // proposed for inclusion in Boost
    #include <iostream>
    #include <cmath>

    int main(int, char**) {
        namespace bh = boost::histogram;

        bh::histogram h(bh::regular_axis(10, -1.0, 2.0, "x"));

        h.fill(-1.5); // put in underflow bin
        h.fill(-1.0); // included, interval is semi-open
        h.fill(-0.5);
        h.fill(1.1);
        h.fill(0.3);
        h.fill(1.7);
        h.fill(2.0);  // put in overflow bin, interval is semi-open
        h.fill(20.0); // put in overflow bin
        h.wfill(0.1, 5.0); // fill with a weighted entry

        for (int i = -1; i <= h.bins(0); ++i) {
            const bh::regular_axis& a = h.axis<bh::regular_axis>(0);
            std::cout << "bin " << i
                      << " x in [" << a[i] << ", " << a[i+1] << "): "
                      << h.value(i) << " +/- " << std::sqrt(h.variance(i))
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

    h = bh.histogram(bh.regular_axis(10, 0.0, 5.0, "radius",
                                     uoflow=False),
                     bh.polar_axis(4, 0.0, "phi"))

    x = np.random.randn(1000)             # generate x
    y = np.random.randn(1000)             # generate y
    rphi = np.empty((1000, 2))
    rphi[:, 0] = (x ** 2 + y ** 2) ** 0.5 # compute radius
    rphi[:, 1] = np.arctan2(y, x)         # compute phi
    h.fill(rphi)

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

The following table shows results of a simple benchmark against

* `TH1I`, `TH3I` and `THnI` of the [ROOT framework](https://root.cern.ch>)

* `histogram` and `histogramdd` from the Python module `numpy`

The benchmark against ROOT is implemented in C++, the benchmark against numpy in Python. For a full discussion of the benchmark, see the section *Notes* in the documentation.

Test system: Intel Core i7-4500U CPU clocked at 1.8 GHz, 8 GB of DDR3 RAM

```
=================  =======  =======  =======  =======  =======  =======
distribution                uniform                    normal
-----------------  -------------------------  -------------------------
dimension          1D       3D       6D       1D       3D       6D
=================  =======  =======  =======  =======  =======  =======
No. of fills       12M      4M       2M       12M      4M       2M
C++: ROOT  [t/s]   0.127    0.199    0.185    0.168    0.143    0.179
C++: boost [t/s]   0.172    0.177    0.155    0.172    0.171    0.150
Py: numpy [t/s]    0.825    0.727    0.436    0.824    0.426    0.401
Py: boost [t/s]    0.209    0.229    0.192    0.207    0.194    0.168
=================  =======  =======  =======  =======  =======  =======
```

`boost::histogram` shows consistent performance comparable to the specialized ROOT histograms. It is faster than ROOT's implementation of a N-dimensional histogram `THnI`. The performance of `boost::histogram` is similar in C++ and Python, showing only a small overhead in Python. It is consistently faster than numpy's histogram functions.

## Rationale

There is a lack of a widely-used free histogram class. While it is easy to write an 1-dimensional histogram, writing an n-dimensional histogram poses more of a challenge. If you add serialization and Python/Numpy support onto the wish-list, the air becomes thin. The main competitor is the [ROOT framework](https://root.cern.ch). This histogram class is designed to be more convenient to use, and as fast or faster than the equivalent ROOT histograms. It comes without heavy baggage, instead it has a clean and modern C++ design which follows the advice given in popular C++ books, like those of [Meyers](http://www.aristeia.com/books.html) and [Sutter and Alexandrescu](http://www.gotw.ca/publications/c++cs.htm).

## Design choices

I designed the histogram based on a decade of experience collected in working with Big Data, more precisely in the field of particle physics and astroparticle physics. I follow these principles:

* "Do one thing and do it well", Doug McIlroy
* The [Zen of Python](https://www.python.org/dev/peps/pep-0020) (also applies to other languages)

### Interface convenience, language transparency

A histogram should have the same consistent interface whatever the dimension. Like `std::vector` it should *just work*, users shouldn't be forced to make *a priori* choices among several histogram classes and options everytime they encounter a new data set.

Python is a great language for data analysis, so the histogram needs Python bindings. Data analysis in Python is Numpy-based, so Numpy support is a must. The histogram should be usable as an interface between a complex simulation or data-storage system written in C++ and data-analysis/plotting in Python: define the histogram in Python, let it be filled on the C++ side, and then get it back for further data analysis or plotting. 

### Powerful binning strategies

The histogram supports about half a dozent different binning strategies, conveniently encapsulated in axis objects. There is the standard sorting of real-valued data into bins of equal or varying width, but also binning of angles or integer values.

Extra bins that count over- and underflow values are added by default. This feature can be turned off individually for each dimension to conserve memory. The extra bins do not disturb normal counting. On an axis with n-bins, the first bin has the index `0`, the last bin `n-1`, while the under- and overflow bins are accessible at `-1` and `n`, respectively.

### Performance, cache-friendliness and memory-efficiency

Dense storage in memory is a must for high performance. Unfortunately, the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) quickly become a problem as the number of dimensions grows, leading to histograms which consume large amounts (up to GBs) of memory.

Fortunately, having many dimensions typically reduces the number of counts per bin, since tuples get spread over many dimensions. The histogram uses an adaptive count size per bin to exploit this, which starts with the smallest size per bin of 1 byte and increases transparently as needed up to 8 byte per bin. A `std::vector` grows in *length* as new elements are added, while the count storage grows in *depth*.

### Support for weighted counts and variance estimates

A histogram categorizes and counts, so the natural choice for the data type of the counts are integers. However, in particle physics, histograms are often filled with weighted events, for example, to make sure that two histograms look the same in one variable, while the distribution of another, correlated variable is a subject of study.

This histogram can be filled with either weighted or unweighted counts. In the weighted case, the sum of weights is stored in a double. The histogram provides a variance estimate is both cases. In the unweighted case, the estimate is computed from the count itself, using Poisson-theory. In the weighted case, the sum of squared weights is stored alongside the sum of weights, and used to compute a variance estimate.

## State of project

The histogram is feature-complete for a 1.0 version. More than 300 individual tests make sure that the implementation works as expected. Comprehensive documentation is available. To grow further, the project needs test users, code review, and feedback.
