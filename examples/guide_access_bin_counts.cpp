//[ guide_access_bin_counts

#include <boost/histogram.hpp>
#include <numeric>
#include <cassert>

namespace bh = boost::histogram;

int main() {
  // make histogram with 2 x 2 = 4 bins (not counting under-/overflow bins)
  auto h = bh::make_histogram(bh::axis::regular<>(2, -1, 1),
                              bh::axis::regular<>(2, 2, 4));

  h(bh::weight(1), -0.5, 2.5); // bin index 0, 0
  h(bh::weight(2), -0.5, 3.5); // bin index 0, 1
  h(bh::weight(3), 0.5, 2.5);  // bin index 1, 0
  h(bh::weight(4), 0.5, 3.5);  // bin index 1, 1

  // access count value, number of indices must match number of axes
  assert(h.at(0, 0) == 1);
  assert(h.at(0, 1) == 2);
  assert(h.at(1, 0) == 3);
  assert(h.at(1, 1) == 4);

  // histogram also supports access via container; using a container of
  // wrong size trips an assertion in debug mode
  auto idx = {0, 1};
  assert(h.at(idx) == 2);

  // histogram also provides bin iterators
  auto sum = std::accumulate(h.begin(), h.end(), 0.0);
  assert(sum == 10);

  // bin iterators are fancy iterators with extra methods
  // (note: iteration order is an implementation detail, don't rely on it)
  std::ostringstream os;
  for (auto it = h.begin(), end = h.end(); it != end; ++it) {
    os << std::setw(2) << it.idx(0) << " " << std::setw(2) << it.idx(1) << ": " << *it;
  }

  assert(os.str() ==
    " 0  0: 1"
    " 1  0: 3"
    " 2  0: 0"
    "-1  0: 0"
    " 0  1: 2"
    " 1  1: 4"
    " 2  1: 0"
    "-1  1: 0"
    " 0  2: 0"
    " 1  2: 0"
    " 2  2: 0"
    "-1  2: 0"
    " 0 -1: 0"
    " 1 -1: 0"
    " 2 -1: 0"
    "-1 -1: 0"
  );
}

//]
