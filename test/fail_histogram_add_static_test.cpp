#include <boost/histogram.hpp>

using namespace boost::histogram;
int main() {
  auto a = make_static_histogram(axis::integer<>(0, 2));
  auto b = make_static_histogram(axis::integer<>(0, 3));
  a += b;
}
