//[ guide_make_dynamic_histogram

#include <boost/histogram.hpp>
#include <vector>

namespace bh = boost::histogram;

int main() {
  // create vector of axes, axis::any is a polymorphic axis type
  auto v = std::vector<bh::axis::any_std>();
  v.push_back(bh::axis::regular<>(100, -1, 1));
  v.push_back(bh::axis::integer<>(1, 7));

  // create dynamic histogram (make_static_histogram cannot be used with iterators)
  auto h = bh::make_dynamic_histogram(v.begin(), v.end());

  // do something with h



  // make_dynamic_histogram copies axis objects; to instead move the whole axis
  // vector into the histogram, create a histogram instance directly
  auto h2 = bh::histogram<decltype(v)>(std::move(v));

  // do something with h2
}

//]
