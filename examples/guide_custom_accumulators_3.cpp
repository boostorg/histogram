// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_custom_accumulators_3

#include <boost/format.hpp>
#include <boost/histogram.hpp>
#include <iostream>

int main() {
  using namespace boost::histogram;

  // Make a accumulator which accepts two samples and computes the mean of each.
  struct multi_mean {
    accumulators::mean<> mean_x, mean_y;

    void operator()(double x, double y) {
      mean_x(x);
      mean_y(y);
    }
  };

  // Create a histogram that uses the custom accumulator.
  auto h = make_histogram_with(dense_storage<multi_mean>(), axis::integer<>(0, 2));
  h(0, sample(1, 2)); // samples go to first cell
  h(0, sample(3, 4)); // samples go to first cell
  h(1, sample(5, 6)); // samples go to second cell
  h(1, sample(7, 8)); // samples go to second cell

  std::ostringstream os;
  for (auto&& x : indexed(h)) {
    os << boost::format("index %i mean_x %.1f mean_y %.1f\n") % x.index() %
              x->mean_x.value() % x->mean_y.value();
  }
  std::cout << os.str() << std::flush;
  assert(os.str() == "index 0 mean_x 2.0 mean_y 3.0\n"
                     "index 1 mean_x 6.0 mean_y 7.0\n");
}

//]
