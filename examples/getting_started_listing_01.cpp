// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ getting_started_listing_01

#include <algorithm>
#include <boost/histogram.hpp>
#include <cassert>
#include <functional>
#include <sstream>

int main() {
  using namespace boost::histogram;

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
    Iterate over bins with a fancy histogram iterator
    - order in which bins are iterated over is an implementation detail
    - iterator dereferences to histogram::const_reference, which is defined by
      its storage class; for the default storage it is actually a plain double
    - idx(N) method returns the index of the N-th axis
    - bin(N_c) method returns current bin of N-th axis; the suffx _c turns
      the argument into a compile-time number, which is needed to return
      a different `bin_type`s for each axis
    - `bin_type` usually is a semi-open interval representing the bin, whose
      edges can be accessed with methods `lower()` and `upper()`, but the
      implementation depends on the axis, please look it up in the reference
  */
  std::ostringstream os;
  os.setf(std::ios_base::fixed);
  for (auto b : indexed(h)) {
    const auto idx = b.first[0];
    const auto interval = h.axis()[idx];
    os << "bin " << std::setw(2) << idx << " [" << std::setprecision(1) << std::setw(4)
       << interval.lower() << ", " << std::setw(4) << interval.upper()
       << "): " << b.second << "\n";
  }

  std::cout << os.str() << std::endl;

  assert(os.str() == "bin  0 [-1.0, -0.5): 1.0\n"
                     "bin  1 [-0.5, -0.0): 1.0\n"
                     "bin  2 [-0.0,  0.5): 2.0\n"
                     "bin  3 [ 0.5,  1.0): 0.0\n"
                     "bin  4 [ 1.0,  1.5): 1.0\n"
                     "bin  5 [ 1.5,  2.0): 1.0\n"
                     "bin  6 [ 2.0,  inf): 2.0\n"
                     "bin -1 [-inf, -1.0): 1.0\n");
  // note how under- and overflow bins appear at the end
}

//]
