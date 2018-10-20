//[ guide_custom_minimal_axis

#include <boost/histogram.hpp>
#include <cassert>

namespace bh = boost::histogram;

// stateless axis which returns 1 if the input is even and 0 otherwise
struct minimal_axis {
  int operator()(int x) const { return x % 2 == 0; }
  unsigned size() const { return 2; }
};

int main() {
  auto h = bh::make_histogram(minimal_axis());

  h(0); h(1); h(2);

  assert(h.at(0) == 2); // two even numbers
  assert(h.at(1) == 1); // one uneven number
}

//]
