//[ guide_custom_storage

#include <array>
#include <boost/histogram.hpp>
#include <vector>

namespace bh = boost::histogram;

int main() {
  // create static histogram with vector<int> as counter storage
  auto h1 = bh::make_histogram_with(std::vector<int>(), bh::axis::regular<>(10, 0, 1));

  // create static histogram with array<int, 12> as counter storage (no allocation!)
  auto h2 = bh::make_histogram_with(std::array<int, 12>(), bh::axis::regular<>(10, 0, 1));

  // do something with h1 and h2
}

//]
