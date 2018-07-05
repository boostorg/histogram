#include <boost/histogram.hpp>
#include <vector>

using namespace boost::histogram;
int main() {
  auto h = make_dynamic_histogram(axis::integer<>(0, 2));
  struct non_convertible_to_int {};
  h.at(non_convertible_to_int{});
}
