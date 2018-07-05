#include <boost/histogram.hpp>

using namespace boost::histogram;
int main() {
  auto h = make_dynamic_histogram(axis::integer<>(0, 2));
  h.at(-2);
}
