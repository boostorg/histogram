//[ guide_make_static_histogram

#include <boost/histogram.hpp>
#include <cassert>

namespace bh = boost::histogram;

int main() {
  /*
      create a static 1d-histogram in default configuration
      which covers the real line from -1 to 1 in 100 bins
  */
  auto h = bh::make_histogram(bh::axis::regular<>(100, -1, 1));
  assert(h.rank() == 1);
}

//]
