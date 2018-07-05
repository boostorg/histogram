#include <boost/histogram.hpp>
#include <vector>

using namespace boost::histogram;
int main() {
  auto h = make_dynamic_histogram(axis::integer<>(0, 2),
                                  axis::integer<>(0, 2));
  h.at(std::vector<int>({-2, 0}));
}
