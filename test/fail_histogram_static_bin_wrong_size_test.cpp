#include <boost/histogram.hpp>

using namespace boost::histogram;
int main() {
  auto h = make_static_histogram(axis::integer<>(0, 2));
  h.at(std::vector<int>({0, 0}));
}
