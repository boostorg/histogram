//[ guide_axis_with_uoflow_off

#include <boost/histogram.hpp>

namespace bh = boost::histogram;

int main() {
  // create a 1d-histogram for dice throws with integer values from 1 to 6
  auto h = bh::make_static_histogram(
      bh::axis::integer<>(1, 7, "eyes", bh::axis::uoflow_type::off));

  // do something with h
}

//]
