// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_fill_accumulator

#include <boost/format.hpp>
#include <boost/histogram.hpp>
#include <cassert>
#include <iostream>
#include <sstream>
#include <utility>

int main() {
  using namespace boost::histogram;

  // make a profile, it computes the mean of the samples in each histogram cell
  auto h = make_profile(axis::regular<>(3, 0.0, 1.0));

  // mean is computed from the values marked with the sample() helper function
  h(0.10, sample(2.5)); // 2.5 goes to bin 0
  h(0.15, sample(3.5)); // 3.5 goes to bin 0
  h(0.25, sample(1.2)); // 1.2 goes to bin 1
  h(0.31, sample(3.4)); // 3.4 goes to bin 1

  // fills from tuples are also supported
  auto xs = std::make_tuple(0.5, sample(3.1));
  h(xs);

  // builtin accumulators have methods to access their state
  std::ostringstream os;
  for (auto x : indexed(h)) {
    // use `.` to access methods of accessor, like `index()`
    // use `->` to access methods of accumulator
    const auto i = x.index();
    const auto n = x->count();     // how many samples are in this bin
    const auto vl = x->value();    // mean value
    const auto vr = x->variance(); // estimated variance of the mean value
    os << boost::format("bin %i count %i value %.1f variance %.1f\n") % i % n % vl % vr;
  }

  std::cout << os.str() << std::flush;

  // assert(os.str() == "bin 0 count 2 value 3.0 variance 0.2\n"
  //                    "bin 1 count 1 value 3.0 variance 0.2\n"
  //                    "bin 2 count 1 value 3.0 variance 0.2\n"
  //                    "bin 3 count 0 value 3.0 variance 0.2\n"
  //                    "bin 4 count 0 value 3.0 variance 0.2\n");
}

//]
