# Histogram

**Fast multi-dimensional histogram with convenient interface for C++11 and Python**

Branch  | Linux [1] and OSX [2] | Windows [3] | Coverage
------- | --------------------- |------------ | --------
master  | [![Build Status Travis](https://travis-ci.org/HDembinski/histogram.svg?branch=master)](https://travis-ci.org/HDembinski/histogram?branch=master) | [![Build status Appveyor](https://ci.appveyor.com/api/projects/status/6a15ga3upiv9ca51/branch/master?svg=true)](https://ci.appveyor.com/project/HDembinski/histogram/branch/master) | [![Coverage Status](https://coveralls.io/repos/github/HDembinski/histogram/badge.svg?branch=master&service=github)](https://coveralls.io/github/HDembinski/histogram?branch=master)
develop | [![Build Status Travis](https://travis-ci.org/HDembinski/histogram.svg?branch=develop)](https://travis-ci.org/HDembinski/histogram?branch=develop) | [![Build status Appveyor](https://ci.appveyor.com/api/projects/status/6a15ga3upiv9ca51/branch/develop?svg=true)](https://ci.appveyor.com/project/HDembinski/histogram/branch/develop) | [![Coverage Status](https://coveralls.io/repos/github/HDembinski/histogram/badge.svg?branch=develop&service=github)](https://coveralls.io/github/HDembinski/histogram?branch=develop)

1. gcc-4.8.4, clang-5.0.0, Python-2.7 & 3.6
2. Xcode 8.3, Python-2.7
3. Visual Studio 14 2015


This `C++11` library provides a multi-dimensional [histogram](https://en.wikipedia.org/wiki/Histogram) class for your statistics needs. The library is **header-only**, if you don't need the Python module.

The histogram is very customisable through policy classes, but the default policies were carefully designed so that most users don't need to customize anything. In the standard configuration, this library offers a unique safety guarantee not found elsewhere: bin counts *cannot overflow* or *be capped*. While being safe to use, the library also has a convenient interface, is memory conserving, and faster than other libraries (see benchmarks).

The histogram class comes in two variants which share a common interface. The *static* variant uses compile-time information to provide maximum performance, at the cost of runtime flexibility and potentially larger executables. The *dynamic* variant is a bit slower, but configurable at run-time and may produce smaller executables. Python bindings for the latter are included, implemented with [Boost.Python](https://www.boost.org/doc/libs/release/libs/python/).

The histogram supports value semantics. Histograms can be added and scaled. Move operations and trips over the language boundary from C++ to Python and back are cheap. Histogram instances can be streamed from/to files and pickled in Python. [Numpy](http://www.numpy.org) is supported to speed up operations in Python: histograms can be filled with Numpy arrays at high speed (in most cases several times faster than numpy's own histogram function) and are convertible into Numpy array views without copying data.

My goal is to submit this project to [Boost](http://www.boost.org), that's why it uses the Boost directory structure and namespace. The code is released under the [Boost Software License](http://www.boost.org/LICENSE_1_0.txt).

Check out the [full documentation](http://hdembinski.github.io/histogram/doc/html/). Highlights are given below.

## Features

* Multi-dimensional histogram
* Simple and convenient interface in C++ and Python
* Static and dynamic implementation in C++ with common interface
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
* Support for addition and scaling of histograms
* Optional serialization based on [Boost.Serialization](https://www.boost.org/doc/libs/release/libs/serialization/)
* Optional Python-bindings that work with [Python-2.7 to 3.6](http://www.python.org)  with [Boost.Python](https://www.boost.org/doc/libs/release/libs/python/)
* Optional [Numpy](http://www.numpy.org) support

(+) In the standard configuration, if you don't use weighted increments.
(++) If you don't use weighted increments, variance tracking come at zero cost. If you use weighted increments, extra space is reserved internally to keep track of a variance counter per bin. The conversion happens automatically and transparently.

## Dependencies

* [Boost >= 1.66](http://www.boost.org) *header-only installation*
* Optional: [CMake >= 3.2](https://cmake.org) [Python >= 2.7](http://www.python.org) [Numpy](http://www.numpy.org) [Boost.Python](https://www.boost.org/doc/libs/release/libs/python/) [Boost.Serialization](https://www.boost.org/doc/libs/release/libs/serialization/) [Boost.Iostreams](https://www.boost.org/doc/libs/release/libs/iostreams/)

## Build instructions

If you don't need the Python module and don't want to run the tests, there is nothing to build. Just copy the content of the include folder to a place where your project can find it.

The Python module and the tests can be build with `b2` from the Boost project or `cmake`. If you are not a Boost developer, use `cmake`.

```sh
git clone https://github.com/HDembinski/histogram.git
mkdir build && cd build
cmake ../histogram/build
make # or 'make install'
```

To run the tests, do `make test`.

### Trouble-shooting when compiling with Python support

If you compile the library with Python support (the default, when you compile at all) and have several versions of Python installed, `cmake` will use the Python interpreter that is accessible as `python` on your system, and will use the headers and libraries of this version. Please make sure that this is the same version that Boost.Python and Boost.Numpy were compiled against, otherwise you will get strange errors during compilation and/or at runtime! You can force `cmake` to pick a specific Python interpreter with the PYTHON_EXECUTABLE flag. For example, to force the use of `python3`, do: `cmake ../histogram/build -DPYTHON_EXECUTABLE=python3`

If you installed Boost with `brew` on OSX and have trouble with the build, have a look at this [Stackoverflow](https://stackoverflow.com/questions/33653001/unable-to-link-against-boost-python-on-os-x) question.

## Code examples

For the full version of the following examples with explanations, see
the [Getting started](http://hdembinski.github.io/histogram/doc/html/histogram/getting_started.html) section in the documentation.

Example 1: Fill a 1d-histogram in C++

```cpp

int main(int, char**) {
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

Example 2: Fill a 2d-histogram in Python with data in Numpy arrays

```python
    import histogram as bh
    import numpy as np

    h = bh.histogram(bh.axis.regular(10, 0.0, 5.0, "radius", uoflow=False),
                     bh.axis.circular(4, 0.0, 2 * np.pi, "phi"))

    x = np.random.randn(1000)             # generate x
    y = np.random.randn(1000)             # generate y
    radius = (x ** 2 + y ** 2) ** 0.5
    phi = np.arctan2(y, x)

    h(radius, phi)

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

Thanks to meta-programming and dynamic memory management, this library is not only safer, more flexible and convenient to use, but also faster than the competition. In the plot below, its speed is compared to classes from the [GNU Scientific Library](https://www.gnu.org/software/gsl), the [ROOT framework from CERN](https://root.cern.ch), and to the histogram functions in [Numpy](http://www.numpy.org). The orange to red items are different compile-time configurations of the histogram in this library. More details on the benchmark are given in the [documentation](http://hdembinski.github.io/histogram/doc/html/histogram/benchmarks.html)

![alt benchmark](doc/benchmark.png)

## What users say

**John Buonagurio** | Manager at [**E<sup><i>x</i></sup>ponent<sup>&reg;</sup>**](www.exponent.com)

*"I just wanted to say 'thanks' for your awesome Histogram library. I'm working on a software package for processing meteorology data and I'm using it to generate wind roses with the help of Qt and QwtPolar. Looks like you thought of just about everything here &ndash; the circular axis type was practically designed for this application, everything 'just worked'."*

## Rationale

There is a lack of a widely-used free histogram class in C++. While it is easy to write a one-dimensional histogram, writing a general multi-dimensional histogram is not trivial. Even more so, if you want the histogram to be serializable and have Python-bindings and support Numpy. In high-energy physics, the [ROOT framework](https://root.cern.ch) from CERN is widely used. This histogram class is designed to be more convenient, flexible, and faster than the equivalent ROOT histograms. It is easy to integrate in your project without adding a huge dependency; you only need Boost. Finally, this library comes in a clean and modern C++11 design which follows the STL and Boost styles, and the general advice given by popular C++ experts ([Meyers](http://www.aristeia.com/books.html), [Sutter and Alexandrescu](http://www.gotw.ca/publications/c++cs.htm), and others).

Read more about the design choices in the [documentation](http://hdembinski.github.io/histogram/doc/html/histogram/rationale.html)

## State of project

The histogram is feature-complete. More than 500 individual tests make sure that the implementation works as expected. Full documentation is available. User feedback is appreciated!

We are finalising the interface for the review process, so code-breaking changes of the interface are currently happening on the master. If you want to use the library in production code, please use the [latest release](https://github.com/HDembinski/histogram/releases) instead of the master. After the library is accepted as part of Boost, the interface will be kept stable, of course.

Review of the library is planned in September 2018.
