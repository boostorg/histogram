#include <boost/histogram.hpp>
#include <tuple>

using namespace boost::histogram;
int main() {
  auto h = make_static_histogram(axis::integer<>(0, 2), axis::integer<>(0, 2));
  h.at(std::make_tuple(0, -2));
}
