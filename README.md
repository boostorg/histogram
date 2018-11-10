# Histogram

**Fast multi-dimensional histogram with convenient interface for C++14 and Python**

Branch  | Linux [1] and OSX [2] | Windows [3] | Coverage
------- | --------------------- |------------ | --------
master  | [![Build Status Travis](https://travis-ci.org/HDembinski/histogram.svg?branch=master)](https://travis-ci.org/HDembinski/histogram?branch=master) | [![Build status Appveyor](https://ci.appveyor.com/api/projects/status/6a15ga3upiv9ca51/branch/master?svg=true)](https://ci.appveyor.com/project/HDembinski/histogram/branch/master) | [![Coverage Status](https://coveralls.io/repos/github/HDembinski/histogram/badge.svg?branch=master&service=github)](https://coveralls.io/github/HDembinski/histogram?branch=master)
develop | [![Build Status Travis](https://travis-ci.org/HDembinski/histogram.svg?branch=develop)](https://travis-ci.org/HDembinski/histogram?branch=develop) | [![Build status Appveyor](https://ci.appveyor.com/api/projects/status/6a15ga3upiv9ca51/branch/develop?svg=true)](https://ci.appveyor.com/project/HDembinski/histogram/branch/develop) | [![Coverage Status](https://coveralls.io/repos/github/HDembinski/histogram/badge.svg?branch=develop&service=github)](https://coveralls.io/github/HDembinski/histogram?branch=develop)

1. gcc-5.5.0, clang-5.0.0
2. Xcode 9.4
3. Visual Studio 15 2017


This `C++14` **header-only** open-source library provides a state-of-the-art multi-dimensional [histogram](https://en.wikipedia.org/wiki/Histogram) class for the professional statistician and everyone who needs to counts things. Actually, this histogram can do more than counting. It can be equipped with arbitrary accumulators to compute means, medians, and whatever you fancy in each cell. Several parallelization options are provided. Check out the [full documentation](http://hdembinski.github.io/histogram/doc/html/). [Python bindings](https://github.com/hdembinski/histogram-python) to this library are available elsewhere.

The histogram is very customisable through a templated modular design, but the default options were carefully chosen so that most users don't need to customize anything. It is easy to use for the casual user, but does not restrict the power-user. In the standard configuration, this library offers a unique safety guarantee not found elsewhere: bin counts *cannot overflow* or *be capped*. While being safe to use, the library also has a convenient interface, is memory conserving, and faster than other libraries (see benchmarks).

The histogram class can be configured in several variants from fully *static* to fully *dynamic*. *Static* variants provides more preformance, at the cost of runtime flexibility and potentially larger executables. *Dynamic* variants are slower, but fully configurable at run-time and may produce smaller executables. A dynamic variant is used in the [Python bindings](https://github.com/hdembinski/histogram-python) to this library.

This project was developed for inclusion in [Boost](http://www.boost.org) and passed Boost review in September 2018. The plan is to have a first official Boost-release in April 2019 with the upcoming version 1.70. Of course, you can use it already now. The source code is licensed under the [Boost Software License](http://www.boost.org/LICENSE_1_0.txt).

Check out the [full documentation](http://hdembinski.github.io/histogram/doc/html/). Highlights are given below.

## Features

* Extremely customizable multi-dimensional histogram
* Simple, convenient, STL and Boost-compatible interface
* Static and dynamic implementations under a common interface
* Counters with high dynamic range, cannot overflow or be capped (+)
* Better performance than other libraries (see benchmarks for details)
* Efficient use of memory (+)
* Efficient template meta-programming using C++14 features and [Boost.Mp11](https://www.boost.org/doc/libs/release/libs/mp11/)
* Support for custom mappings of input values to bin indices
* Support for under-/overflow bins (can be disabled to reduce memory consumption)
* Support for weighted increments
* Support for custom accumulators in each histogram cell (++)
* Support for variance estimates (+++)
* Support for completely stack-based histograms 
* Support for adding and scaling histograms
* Support for custom allocators
* Optional serialization based on [Boost.Serialization](https://www.boost.org/doc/libs/release/libs/serialization/)

(+) In the standard configuration, if you don't use weighted increments. The counter capacity is increased dynamically as the cell counts grow. When even the largest plain integral type would overflow, the storage switches to a [Boost.Multiprecision](https://www.boost.org/doc/libs/release/libs/multiprecision/) integer, which is only limited by available memory.
(++) The histogram can be configured to hold an arbitrary accumulator in each cell instead of a simple counter. Extra values can be passed to the histogram, for example, to compute the mean and variance of values which fall into the same cell.
(+++) Variance estimates are useful when histograms are to be compared quantitatively and if a statistical model is fitted to the cell-counts.

## Dependencies

* [Boost >= 1.66](http://www.boost.org) *header-only installation*
* Optional: [CMake >= 3.5](https://cmake.org) [Boost.Serialization](https://www.boost.org/doc/libs/release/libs/serialization/)

## Build instructions

If you don't want to run the tests, there is nothing to build. Just copy the content of the include folder to a place where your project can find it.

The tests can be build with `b2` from the Boost project or `cmake`. If you are not a Boost developer, use `cmake`.

```sh
git clone https://github.com/HDembinski/histogram.git
mkdir build && cd build
cmake ..
make
```

To run the tests, do `make test` or `ctest -v` for more output.

## Code example

The following stripped-down example was taken from the [Getting started](http://hdembinski.github.io/histogram/doc/html/histogram/getting_started.html) section in the documentation. Have a look into the docs to see the full version with comments and more examples.

Example: Fill a 1d-histogram

```cpp
#include <boost/histogram.hpp>
#include <functional> // for std::ref

int main() {
    namespace bh = boost::histogram;
    using namespace bh::literals; // enables _c suffix

    auto h = bh::make_histogram(
      bh::axis::regular<>(6, -1.0, 2.0, "x")
    );

    // fill histogram
    auto data = { -0.4, 1.1, 0.3, 1.7 };
    std::for_each(data.begin(), data.end(), std::ref(h));

    // iterate over bins
    for (auto it = h.begin(); it != h.end(); ++it) {
      const auto bin = it.bin(0_c);
      std::cout << "bin " << it.idx(0) << " x in ["
                << bin.lower() << ", " << bin.upper() << "): "
                << *it << std::endl;
    }

    /* program output: (note that under- and overflow bins appear at the end)

    bin 0 x in [-1.0, -0.5): 0
    bin 1 x in [-0.5,  0.0): 1
    bin 2 x in [ 0.0,  0.5): 1
    bin 3 x in [ 0.5,  1.0): 0
    bin 4 x in [ 1.0,  1.5): 1
    bin 5 x in [ 1.5,  2.0): 1
    bin 6 x in [ 2.0, inf): 0
    bin -1 x in [-inf, -1): 0

    */
}
```

## Benchmarks

Thanks to meta-programming and dynamic memory management, this library is not only safer, more flexible and convenient to use, but also faster than the competition. In the plot below, its speed is compared to classes from the [GNU Scientific Library](https://www.gnu.org/software/gsl), the [ROOT framework from CERN](https://root.cern.ch), and to the histogram functions in [Numpy](http://www.numpy.org). The orange to red items are different compile-time configurations of the histogram in this library. More details on the benchmark are given in the [documentation](http://hdembinski.github.io/histogram/doc/html/histogram/benchmarks.html)

![alt benchmark](doc/benchmark.png)

## What users say

**John Buonagurio** | Manager at [**E<sup><i>x</i></sup>ponent<sup>&reg;</sup>**](www.exponent.com)

*"I just wanted to say 'thanks' for your awesome Histogram library. I'm working on a software package for processing meteorology data and I'm using it to generate wind roses with the help of Qt and QwtPolar. Looks like you thought of just about everything here &ndash; the circular axis type was practically designed for this application, everything 'just worked'."*

## Rationale

There is a lack of a widely-used free histogram class in C++. While it is easy to write a one-dimensional histogram, writing a general multi-dimensional histogram is not trivial. In high-energy physics, the [ROOT framework](https://root.cern.ch) from CERN is widely used, but it comes with a large dependency. This histogram class is designed to be more convenient, flexible, and faster than the equivalent ROOT histograms. It is easy to integrate in your project if you already use Boost. The library comes in a C++14 design which follows the STL and Boost styles, and the general advice given by well-respected C++ experts ([Meyers](http://www.aristeia.com/books.html), [Sutter and Alexandrescu](http://www.gotw.ca/publications/c++cs.htm), [Stroustrup](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) and others).

Read more about the design choices in the [documentation](http://hdembinski.github.io/histogram/doc/html/histogram/rationale.html)

## State of project

The histogram is nearly feature-complete. More than 500 individual tests make sure that the implementation works as expected. Full documentation is available. User feedback is appreciated!

We are finalising the interface for the review process, so code-breaking changes of the interface are currently happening on the master. If you want to use the library in production code, please use the [latest release](https://github.com/HDembinski/histogram/releases) instead of the master. After the library is accepted as part of Boost, the interface will be kept stable, of course.

Review of the library is planned in September 2018.
