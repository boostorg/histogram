//[ guide_histogram_reduction

#include <boost/histogram.hpp>
#include <iostream>

namespace bh = boost::histogram;

// example of a generic function for histograms, this one sums all entries
template <typename... Ts>
typename bh::histogram<Ts...>::element_type sum(const bh::histogram<Ts...>& h) {
    auto result = typename bh::histogram<Ts...>::element_type(0);
    for (auto x : h)
        result += x;
    return result;
}

int main() {
    using namespace bh::literals; // enables _c suffix

    // make a 2d histogram
    auto h = bh::make_static_histogram(bh::axis::regular<>(3, -1, 1),
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
    std::cout << sum(h).value() << " "
              << sum(hr0).value() << " "
              << sum(hr1).value() << std::endl;
    // prints: 3 3 3

    for (auto yi : h.axis(1_c)) {
        for (auto xi : h.axis(0_c)) {
            std::cout << h.at(xi, yi).value() << " ";
        }
        std::cout << std::endl;
    }
    // prints: 1 0 0
    //         0 0 0
    //         0 1 0
    //         0 0 1

    for (auto xi : hr0.axis())
        std::cout << hr0.at(xi).value() << " ";
    std::cout << std::endl;
    // prints: 1 1 1

    for (auto yi : hr1.axis())
        std::cout << hr1.at(yi).value() << " ";
    std::cout << std::endl;
    // prints: 1 0 1 1
}

//]
