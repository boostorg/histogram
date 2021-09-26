#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <iostream>
#include <vector>

namespace bh = boost::histogram;
using uogrowth_t = decltype(bh::axis::option::growth | bh::axis::option::underflow |
                            bh::axis::option::overflow);

using arg_t = boost::variant2::variant<std::vector<double>, double, std::vector<int>, int,
                                       std::vector<std::string>, std::string>;

int main() {
  using axis_type_1 =
      bh::axis::regular<double, bh::use_default, bh::use_default, uogrowth_t>;
  using axis_type_2 = bh::axis::regular<double>;
  using axis_type_3 = bh::axis::integer<int>;
  using axis_type_4 = bh::axis::category<int>;
  using axis_type_5 = bh::axis::category<std::string>;
  using axis_variant =
      bh::axis::variant<axis_type_1, axis_type_2, axis_type_3, axis_type_4, axis_type_5>;

  auto axes_orig = std::vector<axis_variant>({axis_type_1(10, 0, 1)});
  auto h = bh::histogram<std::vector<axis_variant>>(axes_orig);

  std::vector<int> val = {-1};
  auto args = val; // std::vector<std::vector<int>>({val}); // using this instead removes
                   // the crash

  const auto& axes = bh::unsafe_access::axes(h);
  auto vargs = bh::detail::make_stack_buffer<arg_t>(axes);

  bh::detail::for_each_axis(
      axes, [args_it = args.begin(), vargs_it = vargs.begin()](const auto&) mutable {
        const auto& x = *args_it++;
        auto& v = *vargs_it++;
        v = x;
      });

  h.fill(vargs); // CRASH, using h.fill(-1) or h.fill(args) does not crash.

  // std::cout << h << std::endl;
  return 0;
}
