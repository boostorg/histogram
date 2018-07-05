#include <boost/histogram.hpp>
#include <utility>

using namespace boost::histogram;
int main() {
  auto h = make_dynamic_histogram(axis::integer<>(0, 2),
                                  axis::integer<>(0, 2));
  h.at(std::make_pair(-2, 0));
}
