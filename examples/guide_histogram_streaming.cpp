//[ guide_histogram_streaming

#include <boost/histogram.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <sstream>
#include <cassert>

namespace bh = boost::histogram;

int main() {
  namespace axis = bh::axis;

  auto h = bh::make_histogram(
      axis::regular<>(2, -1, 1),
      axis::regular<axis::transform::log<>>(2, 1, 10, "axis 1"),
      axis::circular<double, axis::empty_metadata_type>(4, 0.1, 1.0), // axis without metadata
      axis::variable<>({-1, 0, 1}, "axis 3", axis::option_type::none),
      axis::category<>({2, 1, 3}, "axis 4"),
      axis::integer<>(-1, 1, "axis 5")
    );

  std::ostringstream os;
  os << h;

  std::cout << os.str() << std::endl;

  assert(os.str() == 
    "histogram(\n"
    "  regular(2, -1, 1, options=underflow_and_overflow),\n"
    "  regular_log(2, 1, 10, metadata=\"axis 1\", options=underflow_and_overflow),\n"
    "  circular(4, 0.1, 1.1, options=overflow),\n"
    "  variable(-1, 0, 1, metadata=\"axis 3\", options=none),\n"
    "  category(2, 1, 3, metadata=\"axis 4\", options=overflow),\n"
    "  integer(-1, 1, metadata=\"axis 5\", options=underflow_and_overflow),\n"
    ")"
  );
}

//]
