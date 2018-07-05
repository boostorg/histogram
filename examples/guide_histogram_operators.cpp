//[ guide_histogram_operators

#include <boost/histogram.hpp>
#include <iostream>

namespace bh = boost::histogram;

int main() {
    // make two histograms
    auto h1 = bh::make_static_histogram(bh::axis::regular<>(2, -1, 1));
    auto h2 = bh::make_static_histogram(bh::axis::regular<>(2, -1, 1));

    h1(-0.5);               // counts are: 1 0
    h2(0.5);                // counts are: 0 1

    // add them
    auto h3 = h1;
    h3 += h2;               // counts are: 1 1

    // adding multiple histograms at once is efficient and does not create
    // superfluous temporaries since operator+ functions are overloaded to
    // accept and return rvalue references where possible
    auto h4 = h1 + h2 + h3; // counts are: 2 2

    std::cout << h4.at(0).value() << " " << h4.at(1).value() << std::endl;
    // prints: 2 2

    // multiply by number
    h4 *= 2;                // counts are: 4 4

    // divide by number
    auto h5 = h4 / 4;       // counts are: 1 1

    std::cout << h5.at(0).value() << " " << h5.at(1).value() << std::endl;
    // prints: 1 1

    // compare histograms
    std::cout << (h4 == 4 * h5) << " " << (h4 != h5) << std::endl;
    // prints: 1 1

    // note: special effect of multiplication on counter variance
    auto h = bh::make_static_histogram(bh::axis::regular<>(2, -1, 1));
    h(-0.5); // counts are: 1 0
    std::cout << "value    " << (2 * h).at(0).value()
              << " " << (h + h).at(0).value() << "\n"
              << "variance " << (2 * h).at(0).variance()
              << " " << (h + h).at(0).variance() << std::endl;
    // equality operator also checks variances, so the statement is false
    std::cout << (h + h == 2 * h) << std::endl;
    /* prints:
        value    2 2
        variance 4 2
        0
    */
}

//]
