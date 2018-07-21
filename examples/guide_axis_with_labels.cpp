//[ guide_axis_with_labels

#include <boost/histogram.hpp>

namespace bh = boost::histogram;

int main() {
  // create a 2d-histogram with an "age" and an "income" axis
  auto h = bh::make_static_histogram(
      bh::axis::regular<>(20, 0, 100, "age in years"),
      bh::axis::regular<>(20, 0, 100, "yearly income in $1000"));

  // do something with h
}

//]
