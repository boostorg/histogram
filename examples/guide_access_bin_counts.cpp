// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_access_bin_counts

#include <boost/format.hpp>
#include <boost/histogram.hpp>
#include <cassert>
#include <numeric> // for std::accumulate

using namespace boost::histogram;

int main() {
  // make histogram with 2 x 2 = 4 bins (not counting under-/overflow bins)
  auto h = make_histogram(axis::regular<>(2, -1, 1), axis::regular<>(2, 2, 4));

  h(weight(1), -0.5, 2.5); // bin index 0, 0
  h(weight(2), -0.5, 3.5); // bin index 0, 1
  h(weight(3), 0.5, 2.5);  // bin index 1, 0
  h(weight(4), 0.5, 3.5);  // bin index 1, 1

  // histogram has bin iterators which iterate over all bin values including
  // underflow/overflow; it works with STL algorithms
  auto sum = std::accumulate(h.begin(), h.end(), 0.0);
  assert(sum == 10);

  // access individual bins with `at`
  assert(h.at(0, 0) == 1);
  assert(h.at(0, 1) == 2);
  assert(h.at(1, 0) == 3);
  assert(h.at(1, 1) == 4);

  // index tuples are also supported
  auto idx = std::make_tuple(0, 1);
  assert(h.at(idx) == 2);

  // use the `indexed` range adaptor to iterate over all bins;
  // it is not only more convenient but also faster than a hand-crafted loop!
  std::ostringstream os;
  for (auto x : indexed(h)) {
    // x is a special accessor object
    const auto i = x.index(0); // current index along first axis
    const auto j = x.index(1); // current index along second axis
    const auto b0 = x.bin(0);  // current bin interval along first axis
    const auto b1 = x.bin(1);  // current bin interval along second axis
    const auto v = *x;         // "dereference" to get the bin value
    os << boost::format("%i %i [%2i, %i) [%2i, %i): %i\n") % i % j % b0.lower() %
              b0.upper() % b1.lower() % b1.upper() % v;
  }

  std::cout << os.str() << std::flush;

  assert(os.str() == "0 0 [-1, 0) [ 2, 3): 1\n"
                     "1 0 [ 0, 1) [ 2, 3): 3\n"
                     "0 1 [-1, 0) [ 3, 4): 2\n"
                     "1 1 [ 0, 1) [ 3, 4): 4\n");

  // `indexed` skips underflow and overflow bins by default, but can be called with the
  // second argument `coverage::all` to walk over all bins
  std::ostringstream os2;
  for (auto x : indexed(h, coverage::all)) {
    os2 << boost::format("%2i %2i: %i\n") % x.index(0) % x.index(1) % *x;
  }

  std::cout << os2.str() << std::flush;

  assert(os2.str() == "-1 -1: 0\n"
                      " 0 -1: 0\n"
                      " 1 -1: 0\n"
                      " 2 -1: 0\n"

                      "-1  0: 0\n"
                      " 0  0: 1\n"
                      " 1  0: 3\n"
                      " 2  0: 0\n"

                      "-1  1: 0\n"
                      " 0  1: 2\n"
                      " 1  1: 4\n"
                      " 2  1: 0\n"

                      "-1  2: 0\n"
                      " 0  2: 0\n"
                      " 1  2: 0\n"
                      " 2  2: 0\n");
}

//]
