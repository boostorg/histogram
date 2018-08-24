//[ guide_fill_histogram

#include <boost/histogram.hpp>
#include <utility>
#include <vector>

namespace bh = boost::histogram;

int main() {
  auto h = bh::make_static_histogram(bh::axis::regular<>(8, 0, 4),
                                     bh::axis::regular<>(10, 0, 5));

  // fill histogram, number of arguments must be equal to number of axes
  h(0, 1.1);                // increases bin counter by one
  h(bh::weight(2), 3, 3.4); // increase bin counter by 2 instead of 1

  // histogram also supports fills from a container of values; a container
  // of wrong size trips an assertion in debug mode
  auto xy1 = std::make_pair(4, 3.1);
  h(xy1);
  auto xy2 = std::vector<double>({3.0, 4.9});
  h(xy2);

  // functional-style processing is also supported
  std::vector<std::pair<int, double>> input_data{
      {0, 1.2}, {2, 3.4}, {4, 5.6}};
  // std::for_each takes the functor by value, thus it potentially makes
  // expensive copies of the histogram, but modern compilers are usually smart
  // enough to avoid the superfluous copies
  auto h2 =
      std::for_each(input_data.begin(), input_data.end(),
                    bh::make_static_histogram(bh::axis::regular<>(8, 0, 4),
                                              bh::axis::regular<>(10, 0, 5)));
  // h2 is filled
}

//]
