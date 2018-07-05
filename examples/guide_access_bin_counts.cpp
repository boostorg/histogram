//[ guide_access_bin_counts

#include <boost/histogram.hpp>
#include <iostream>
#include <numeric>

namespace bh = boost::histogram;

int main() {
    // make histogram with 2 x 2 = 4 bins (not counting under-/overflow bins)
    auto h = bh::make_static_histogram(
            bh::axis::regular<>(2, -1, 1),
            bh::axis::regular<>(2,  2, 4)
        );

    h(bh::weight(1), -0.5, 2.5); // bin index 0, 0
    h(bh::weight(2), -0.5, 3.5); // bin index 0, 1
    h(bh::weight(3),  0.5, 2.5); // bin index 1, 0
    h(bh::weight(4),  0.5, 3.5); // bin index 1, 1

    // access count value, number of indices must match number of axes
    std::cout << h.at(0, 0).value() << " "
              << h.at(0, 1).value() << " "
              << h.at(1, 0).value() << " "
              << h.at(1, 1).value()
              << std::endl;

    // prints: 1 2 3 4

    // access count variance, number of indices must match number of axes
    std::cout << h.at(0, 0).variance() << " "
              << h.at(0, 1).variance() << " "
              << h.at(1, 0).variance() << " "
              << h.at(1, 1).variance()
              << std::endl;
    // prints: 1 4 9 16

    // you can also make a copy of the type that holds the bin count
    auto c11 = h.at(1, 1);
    std::cout << c11.value() << " " << c11.variance() << std::endl;
    // prints: 4 16

    // histogram also supports access via container; using a container of
    // wrong size trips an assertion in debug mode
    auto idx = {0, 1};
    std::cout << h.at(idx).value() << std::endl;
    // prints: 2

    // histogram also provides extended iterators
    auto sum = std::accumulate(h.begin(), h.end(),
                               typename decltype(h)::element_type(0));
    std::cout << sum.value() << " " << sum.variance() << std::endl;
    // prints: 10 30
}

//]
