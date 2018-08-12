//[ guide_custom_storage

#include <boost/histogram.hpp>
#include <boost/histogram/storage/array_storage.hpp>

namespace bh = boost::histogram;

int main() {
  // create static histogram with array_storage, using int as counter type
  auto h = bh::make_static_histogram_with(bh::array_storage<int>(),
                                          bh::axis::regular<>(10, 0, 1));

  // do something with h
}

//]
