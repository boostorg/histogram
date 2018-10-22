//[ guide_histogram_reduction

#include <boost/histogram.hpp>
#include <sstream>
#include <cassert>

namespace bh = boost::histogram;

// example of a generic function for histograms, this one sums all entries
template <typename... Ts>
typename bh::histogram<Ts...>::element_type sum(
    const bh::histogram<Ts...>& h) {
  auto result = typename bh::histogram<Ts...>::element_type(0);
  for (auto x : h) result += x;
  return result;
}

int main() {
  using namespace bh::literals; // enables _c suffix

  // make a 2d histogram
  auto h = bh::make_histogram(bh::axis::regular<>(3, -1, 1),
                              bh::axis::integer<>(0, 4));

  h(-0.9, 0);
  h(0.9, 3);
  h(0.1, 2);

  auto hr0 = h.reduce_to(0_c); // keep only first axis
  auto hr1 = h.reduce_to(1_c); // keep only second axis

  /*
      reduce does not remove counts; returned histograms are summed over
      the removed axes, so h, hr0, and hr1 have same number of total counts
  */
  assert(sum(h) == 3 && sum(hr0) == 3 && sum(hr1) == 3);

  std::ostringstream os1;
  for (auto yi : h.axis(1_c)) {
    for (auto xi : h.axis(0_c)) { os1 << h.at(xi, yi) << " "; }
    os1 << "\n";
  }
  assert(os1.str() == 
         "1 0 0 \n"
         "0 0 0 \n"
         "0 1 0 \n"
         "0 0 1 \n");

  std::ostringstream os2;
  for (auto xi : hr0.axis()) os2 << hr0.at(xi) << " ";
  assert(os2.str() == "1 1 1 ");

  std::ostringstream os3;
  for (auto yi : hr1.axis()) os3 << hr1.at(yi) << " ";
  assert(os3.str() == "1 0 1 1 ");
}

//]
