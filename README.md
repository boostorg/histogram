# Histogram

Fast n-dimensional histogram with convenient interface for C++ and Python

This project contains an easy-to-use powerful n-dimensional histogram class implemented in `C++0x`, optimized for convenience and excellent performance under heavy duty. The histogram has a complete C++ and a [Python](http://www.python.org) interface, and can be moved over the language boundary with ease. [Numpy](http://www.numpy.org) is fully supported; histograms can be filled with Numpy arrays at C speeds and are convertible into Numpy arrays without copying data. Histograms can be streamed from/to files and pickled in Python.

My goal is to submit this project to the [Boost](http://www.boost.org) libraries, that's why it uses the boost directory structure and namespace. The code is released under the [Boost Software License](http://www.boost.org/LICENSE_1_0.txt).

### Dependencies

* [Boost](http://www.boost.org)
* [CMake](https://cmake.org)
* Optional:
  [Python](http://www.python.org)
  [Numpy](http://www.numpy.org)
  [Sphinx](http://www.sphinx-doc.org)

### Features

* N-dimensional histogram
* Intuitive and convenient interface, everything *just works*
* Support for different binning scenarios, including binning of angles
* Support for weighted events, with variance estimates for each bin
* Optional underflow- and overflow-bins
* High-performance through cache-friendly design
* Space-efficient memory storage that dynamically grows as needed
* Serialization support with zero-suppression
* Multi-language support: C++ and Python
* Numpy support

### Build instructions

`git clone git@github.com:HDembinski/histogram.git`

`mkdir build; cd build`

`cmake ../histogram.git/CMake`

`make install` (or just `make` to run the tests)

To run the tests, do `make test` or `ctest -V` for more output.

## Rationale

There is a lack of a widely-used free histogram class. While it is easy to write an 1-dimensional histogram, writing an n-dimensional histogram poses more of a challenge. If you add serialization and Python/Numpy support onto the wish-list, the air becomes thin. The main competitor is the [ROOT framework](https://root.cern.ch). This histogram class is designed to be more convenient to use, and as fast or faster than the equivalent ROOT histograms. It comes without heavy baggage, instead it has a clean and modern C++ design which follows the advice given in popular C++ books, like those of [Meyers](http://www.aristeia.com/books.html) and [Sutter and Alexandrescu](http://www.gotw.ca/publications/c++cs.htm).

## Design choices

I designed the histogram based on a decade of experience collected in working with Big Data, more precisely in the field of particle physics and astroparticle physics. I follow these principles:

* "Do one thing and do it well", Doug McIlroy
* The [Zen of Python](https://www.python.org/dev/peps/pep-0020) (also applies to other languages).

### Interface convenience, language transparency

A histogram should have the same consistent interface whatever the dimension. Like `std::vector` it should *just work*, users shouldn't be forced to make *a priori* choices among several histogram classes and options everytime they encounter a new data set. Python is a great language for data analysis, so the histogram needs Python bindings.

Data analysis in Python is Numpy-based, so Numpy support is a must. The histogram should be usable as an interface between a complex simulation or data-storage system written in C++ and data-analysis/plotting in Python: define the histogram in Python, let it be filled on the C++ side, and then get it back for further data analysis or plotting. 

### Powerful binning strategies

The histogram supports half a dozent different binning strategies, conveniently encapsulated in axis objects. There is the standard sorting of real-valued data into bins of equal or varying width, but also binning of angles or integer values.

Extra bins that count over- and underflow values are added by default. This feature can be turned off individually for each dimension. The extra bins do not disturb normal counting. On an axis with n-bins, the first bin has the index `0`, the last bin `n-1`, while the under- and overflow bins are accessible at `-1` and `n`, respectively.

### Performance, cache-friendliness and memory-efficiency

Dense storage in memory is a must for high performance. Unfortunately, the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) quickly become a problem as the number of dimensions grows, leading to histograms which consume large amounts (up to GBs) of memory.

Fortunately, having many dimensions typically reduces the number of counts per bin, since tuples get spread over many dimensions. The histogram uses an adaptive count size per bin to exploit this, which starts with the smallest size per bin of 1 byte and increases transparently as needed up to 8 byte per bin. A `std::vector` grows in *length* as new elements are added, while the count storage grows in *depth*.

### Support for weighted counts and variance estimates

A histogram categorizes and counts, so the natural choice for the data type of the counts are integers. However, in particle physics, histograms are often filled with weighted events, for example, to make sure that two histograms look the same in one variable, while the distribution of another, correlated variable is a subject of study.

This histogram can be filled with either weighted or unweighted counts. In the weighted case, the sum of weights is stored in a double. The histogram provides a variance estimate is both cases. In the unweighted case, the estimate is computed from the count itself, using Poisson-theory. In the weighted case, the sum of squared weights is stored alongside the sum of weights, and used to compute a variance estimate.

## State of project

The histogram is feature-complete for 1.0 version. More than 300 individual tests make sure that the implementation works as expected. Comprehensive documentation is a to-do. To grow further, the project needs test users, code review, and feedback.
