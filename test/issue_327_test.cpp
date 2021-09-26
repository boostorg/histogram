#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <vector>
#include "throw_exception.hpp"

namespace bh = boost::histogram;
using uogrowth_t = decltype(bh::axis::option::growth | bh::axis::option::underflow |
                            bh::axis::option::overflow);

using arg_t = boost::variant2::variant<std::vector<int>, int>;

int main() {
  using axis_type =
      bh::axis::regular<double, bh::use_default, bh::use_default, uogrowth_t>;
  using axis_variant = bh::axis::variant<axis_type>;

  auto axes = std::vector<axis_variant>({axis_type(10, 0, 1)});
  auto h = bh::make_histogram_with(std::vector<int>(), axes);
  BOOST_TEST_EQ(h.rank(), 1);

  std::vector<arg_t> vargs = {-1};
  h.fill(vargs); // CRASH, using h.fill(-1) or h.fill(args) does not crash.

  return boost::report_errors();
}
