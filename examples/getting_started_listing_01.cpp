// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ getting_started_listing_01

#include <algorithm>           // std::for_each
#include <boost/format.hpp>    // only needed for printing
#include <boost/histogram.hpp> // make_histogram, weight, indexed
#include <cassert>             // assert
#include <functional>          // std::ref
#include <sstream>             // std::ostringstream, std::cout, std::flush

int main() {
  using namespace boost::histogram; // strip the boost::histogram prefix

  /*
    Create a 1d-histogram with an axis that has 6 equidistant
    bins on the real line from -1.0 to 2.0, and label it as "x".
  */
  auto h = make_histogram(axis::regular<>(6, -1.0, 2.0, "x"));

  /*
    Fill histogram with data, typically this happens in a loop.
    STL algorithms are supported. std::for_each is very convenient
    to fill a histogram from an iterator range. Make sure to
    use std::ref in the call, otherwise it will fill a copy of
    the histogram and return it, which is less efficient.
  */
  auto data = {-0.5, 1.1, 0.3, 1.7};
  std::for_each(data.begin(), data.end(), std::ref(h));

  /*
    A regular axis is a sequence of semi-open bins. Extra under- and
    overflow bins extend the axis in the default configuration.
    index    :      -1     0     1    2    3    4    5    6
    bin edges:  -inf  -1.0  -0.5  0.0  0.5  1.0  1.5  2.0  inf
  */
  h(-1.5); // put in underflow bin -1
  h(-1.0); // put in bin 0, bin interval is semi-open
  h(2.0);  // put in overflow bin 6, bin interval is semi-open
  h(20.0); // put in overflow bin 6

  /*
    Do a weighted fill using the `weight` function as an additional
    argument. It may appear at the beginning or end of the argument list.
  */
  h(0.1, weight(1.0));

  /*
    Iterate over bins with the `indexed` range adaptor to obtain the current bin index and
    the bin value via a proxy class. By default, under- and overflow bins are skipped.
    Passing `true` as second argument iterates over all bins.
    - Iteration order is implementation defined, `indexed` uses the most efficient one.
    - Access the bin index with operator[] of the proxy, passing the dimension d.
    - Access the corresponding bin interval view with `bin(d)`. Use a compile-time number
      instead of a normal number, if possible, to make this call more performant. The
      return type of this call depends on the axis (see the axis reference for details),
      usually a class that represents a semi-open interval, whose edges can be accessed
      with methods `lower()` and `upper()`.
    - Access the value with the dereference operator. The proxy acts like a pointer to it.
  */
  std::ostringstream os;
  for (auto x : indexed(h, true)) {
    os << boost::format("bin %2i [%4.1f, %4.1f): %i\n") % x[0] % x.bin(0).lower() %
              x.bin(0).upper() % *x;
  }

  std::cout << os.str() << std::flush;

  assert(os.str() == "bin -1 [-inf, -1.0): 1\n"
                     "bin  0 [-1.0, -0.5): 1\n"
                     "bin  1 [-0.5, -0.0): 1\n"
                     "bin  2 [-0.0,  0.5): 2\n"
                     "bin  3 [ 0.5,  1.0): 0\n"
                     "bin  4 [ 1.0,  1.5): 1\n"
                     "bin  5 [ 1.5,  2.0): 1\n"
                     "bin  6 [ 2.0,  inf): 2\n");
  // note how under- and overflow bins appear at the end
}

//]
