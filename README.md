# Histogram

**Fast multi-dimensional histogram with convenient interface for C++11 and Python**

Branch  | Linux [1] and OSX [2] | Windows [3] | Coverage
------- | --------------------- |------------ | --------
master  | [![Build Status Travis](https://travis-ci.org/HDembinski/histogram.svg?branch=master)](https://travis-ci.org/HDembinski/histogram?branch=master) | [![Build status Appveyor](https://ci.appveyor.com/api/projects/status/6a15ga3upiv9ca51/branch/master?svg=true)](https://ci.appveyor.com/project/HDembinski/histogram/branch/master) | [![Coverage Status](https://coveralls.io/repos/github/HDembinski/histogram/badge.svg?branch=master&service=github)](https://coveralls.io/github/HDembinski/histogram?branch=master)
develop | [![Build Status Travis](https://travis-ci.org/HDembinski/histogram.svg?branch=develop)](https://travis-ci.org/HDembinski/histogram?branch=develop) | [![Build status Appveyor](https://ci.appveyor.com/api/projects/status/6a15ga3upiv9ca51/branch/develop?svg=true)](https://ci.appveyor.com/project/HDembinski/histogram/branch/develop) | [![Coverage Status](https://coveralls.io/repos/github/HDembinski/histogram/badge.svg?branch=develop&service=github)](https://coveralls.io/github/HDembinski/histogram?branch=develop)

1. gcc-4.8.4, clang-5.0.0, Python-2.7 & 3.6
2. Xcode 8.3, Python-2.7
3. Visual Studio 14 2015


This `C++11` open-source library provides a state-of-the-art multi-dimensional [histogram](https://en.wikipedia.org/wiki/Histogram) class for the professional statistician and everyone who needs to count things. The library is **header-only**. Check out the [full documentation](http://hdembinski.github.io/histogram/doc/html/). [Python bindings](https://github.com/hdembinski/histogram-python) for this library are available elsewhere.

The histogram is very customisable through policy classes, but the default policies were carefully designed so that most users don't need to customize anything. In the standard configuration, this library offers a unique safety guarantee not found elsewhere: bin counts *cannot overflow* or *be capped*. While being safe to use, the library also has a convenient interface, is memory conserving, and faster than other libraries (see benchmarks).

The histogram class comes in two variants which share a common interface. The *static* variant uses compile-time information to provide maximum performance, at the cost of runtime flexibility and potentially larger executables. The *dynamic* variant is a bit slower, but configurable at run-time and may produce smaller executables. This variant allows one to make dynamic frontends, such as the [Python frontend](https://github.com/hdembinski/histogram-python).

My goal is to submit this project to [Boost](http://www.boost.org), that's why it uses the Boost directory structure and namespace. The code is released under the [Boost Software License](http://www.boost.org/LICENSE_1_0.txt).

Check out the [full documentation](http://hdembinski.github.io/histogram/doc/html/). Highlights are given below.

## Features

* Multi-dimensional histogram
* Simple and convenient interface
* Static and dynamic implementation with common interface
* High dynamic range using [Boost.Multiprecision](https://www.boost.org/doc/libs/release/libs/multiprecision/): Counters cannot overflow or be capped (+)
* Better performance than other libraries (see benchmarks for details)
* Fast compilation thanks to modern template meta-programming using [Boost.Mp11](https://www.boost.org/doc/libs/release/libs/mp11/)
* Efficient move operations
* Efficient conversion between static and dynamic implementation
* Efficient use of memory (counter capacity dynamically grows as needed)
* Support for many mappings of input values to bin indices (user extensible)
* Support for weighted increments
* Support for under-/overflow bins (can be disabled individually for each dimension)
* Support for variance tracking (++)
* Support for adding and scaling histograms
* Support for custom allocators
* Optional serialization based on [Boost.Serialization](https://www.boost.org/doc/libs/release/libs/serialization/)

(+) In the standard configuration, if you don't use weighted increments.
(++) If you don't use weighted increments, variance tracking come at zero cost. If you use weighted increments, extra space is reserved internally to keep track of a variance counter per bin. The conversion happens automatically and transparently.

## Dependencies

* [Boost >= 1.66](http://www.boost.org) *header-only installation*
* Optional: [CMake >= 3.5](https://cmake.org) [Boost.Serialization](https://www.boost.org/doc/libs/release/libs/serialization/)

## Build instructions

If you don't want to run the tests, there is nothing to build. Just copy the content of the include folder to a place where your project can find it.

The tests can be build with `b2` from the Boost project or `cmake`. If you are not a Boost developer, use `cmake`.

```sh
git clone https://github.com/HDembinski/histogram.git
mkdir build && cd build
cmake ../histogram/build
make
```

To run the tests, do `make test` or `ctest -v` for more output.

## Code example

The following stripped-down example was taken from the [Getting started](http://hdembinski.github.io/histogram/doc/html/histogram/getting_started.html) section in the documentation. Have a look into the docs to see the full version with comments and more examples.

Example: Fill a 1d-histogram

```cpp
#include <boost/histogram.hpp>

int main() {
    namespace bh = boost::histogram;
    using namespace bh::literals; // enables _c suffix

    auto h = bh::make_static_histogram(
      bh::axis::regular<>(6, -1.0, 2.0, "x")
    );

    auto data = { -0.4, 1.1, 0.3, 1.7 };
    std::for_each(data.begin(), data.end(), h);

    for (auto it = h.begin(); it != h.end(); ++it) {
      const auto bin = it.bin(0_c);
      std::cout << "bin " << it.idx(0) << " x in ["
                << bin.lower() << ", " << bin.upper() << "): "
                << it->value() << " +/- "
                << std::sqrt(it->variance())
                << std::endl;
    }

    /* program output: (note that under- and overflow bins appear at the end)

    bin 0 x in [-1.0, -0.5): 0 +/- 0
    bin 1 x in [-0.5,  0.0): 1 +/- 1
    bin 2 x in [ 0.0,  0.5): 1 +/- 1
    bin 3 x in [ 0.5,  1.0): 0 +/- 0
    bin 4 x in [ 1.0,  1.5): 1 +/- 1
    bin 5 x in [ 1.5,  2.0): 1 +/- 1
    bin 6 x in [ 2.0, inf): 0 +/- 0
    bin -1 x in [-inf, -1): 0 +/- 0

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

There is a lack of a widely-used free histogram class in C++. While it is easy to write a one-dimensional histogram, writing a general multi-dimensional histogram is not trivial. In high-energy physics, the [ROOT framework](https://root.cern.ch) from CERN is widely used. This histogram class is designed to be more convenient, flexible, and faster than the equivalent ROOT histograms. It is easy to integrate in your project if you already use Boost. The library comes in a modern C++11 design which follows the STL and Boost styles, and the general advice given by popular C++ experts ([Meyers](http://www.aristeia.com/books.html), [Sutter and Alexandrescu](http://www.gotw.ca/publications/c++cs.htm), and others).

Read more about the design choices in the [documentation](http://hdembinski.github.io/histogram/doc/html/histogram/rationale.html)

## State of project

The histogram is feature-complete. More than 500 individual tests make sure that the implementation works as expected. Full documentation is available. User feedback is appreciated!

We are finalising the interface for the review process, so code-breaking changes of the interface are currently happening on the master. If you want to use the library in production code, please use the [latest release](https://github.com/HDembinski/histogram/releases) instead of the master. After the library is accepted as part of Boost, the interface will be kept stable, of course.

Review of the library is planned in September 2018.
