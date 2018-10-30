//[ guide_histogram_operators

#include <boost/histogram.hpp>
#include <cassert>
#include <vector>

namespace bh = boost::histogram;

int main() {
  // make two histograms
  auto h1 = bh::make_histogram(bh::axis::regular<>(2, -1, 1));
  auto h2 = bh::make_histogram(bh::axis::regular<>(2, -1, 1));

  h1(-0.5); // counts are: 1 0
  h2(0.5);  // counts are: 0 1

  // add them
  auto h3 = h1;
  h3 += h2; // counts are: 1 1

  // adding multiple histograms at once is efficient and does not create
  // superfluous temporaries since operator+ functions are overloaded to
  // accept and return rvalue references where possible
  auto h4 = h1 + h2 + h3; // counts are: 2 2

  assert(h4.at(0) == 2 && h4.at(1) == 2);

  // multiply by number
  h4 *= 2; // counts are: 4 4

  // divide by number
  auto h5 = h4 / 4; // counts are: 1 1

  assert(h5.at(0) == 1 && h5.at(1) == 1);
  assert(h4 != h5 && h4 == 4 * h5);

  // note special effect of multiplication on weight_counter variance
  auto h = bh::make_histogram_with(std::vector<bh::weight_counter<double>>(),
                                   bh::axis::regular<>(2, -1, 1));
  h(-0.5);

  // counts are: 1 0
  assert(h.at(0).value() == 1 && h.at(1).value() == 0);

  auto h_sum = h + h;
  auto h_mul = 2 * h;

  // equality operator checks variances, so following statement is false
  assert(h_sum != h_mul);

  // variance is different when histograms are scaled
  assert(h_sum.at(0).value() == 2 && h_mul.at(0).value() == 2);
  assert(h_sum.at(0).variance() == 2 && h_mul.at(0).variance() == 4);
}

//]
