#include <boost/histogram.hpp>

using namespace boost::histogram;
int main() {
  auto a = make_dynamic_histogram(axis::integer<>(0, 2));
  auto b = make_dynamic_histogram(axis::integer<>(0, 3));
  a += b;
}
