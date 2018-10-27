//[ guide_make_dynamic_histogram

#include <boost/histogram.hpp>
#include <vector>
#include <cassert>

namespace bh = boost::histogram;

int main() {
  // create vector of axes, axis::any is a polymorphic axis type
  auto v = std::vector<bh::axis::variant<
      bh::axis::regular<>, bh::axis::integer<>
    >>();
  v.push_back(bh::axis::regular<>(100, -1, 1));
  v.push_back(bh::axis::integer<>(1, 7));

  // create dynamic histogram from iterator range
  auto h = bh::make_histogram(v.begin(), v.end());
  assert(h.rank() == 2);

  // create dynamic histogram by moving the vector (this avoids copies)
  auto h2 = bh::make_histogram(std::move(v));
  assert(h2.rank() == 2);
}

//]
