#include <boost/histogram.hpp>
#include <iostream>

int main(int, char**) {
    namespace bh = boost::histogram;
    using namespace bh::literals; // enables _c suffix

    /*
        create a static 1d-histogram with an axis that has 10 equidistant
        bins on the real line from -1.0 to 2.0, and label it as "x"
    */
    auto h = bh::make_static_histogram(
        bh::axis::regular<>(10, -1.0, 2.0, "x")
    );

    // fill histogram with data, typically this would happen in a loop
    h.fill(-1.5); // put in underflow bin
    h.fill(-1.0); // included in first bin, bin interval is semi-open
    h.fill(-0.5);
    h.fill(1.1);
    h.fill(0.3);
    h.fill(1.7);
    h.fill(2.0);  // put in overflow bin, bin interval is semi-open
    h.fill(20.0); // put in overflow bin

    /*
        use bh::count(N) if you would otherwise call h.fill(...) with
        *same* argument N times, N is an integer argument
    */
    h.fill(1.0, bh::count(4));

    /*
        do a weighted fill using bh::weight, which accepts a double
        - don't mix this with bh::count, both have a different effect on the
          variance (see Rationale for an explanation regarding weights)
        - if you don't know what this is good for, use bh::count instead,
          it is most likeliy what you want and it is more efficient
    */
    h.fill(0.1, bh::weight(2.5));

    /*
        iterate over bins, loop excludes under- and overflow bins
        - index 0_c is a compile-time number, the only way in C++ to make
          axis(...) to return a different type for each index
        - for-loop yields instances of `std::pair<int, bin_type>`, where
          `bin_type` usually is a semi-open interval representing the bin,
          whose edges can be accessed with methods `lower()` and `upper()`,
          but the [bin type] depends on the axis, look it up in the reference
        - `value(index)` method returns the bin count at index
        - `variance(index)` method returns a variance estimate of the bin
          count at index (see Rationale section for what this means)
    */
    for (const auto& bin : h.axis(0_c)) {
        std::cout << "bin " << bin.first << " x in ["
                  << bin.second.lower() << ", " << bin.second.upper() << "): "
                  << h.value(bin.first) << " +/- "
                  << std::sqrt(h.variance(bin.first))
                  << std::endl;
    }

    // accessing under- and overflow bins is easy, use indices -1 and 10
    std::cout << "underflow bin [" << h.axis(0_c)[-1].lower()
              << ", " << h.axis(0_c)[-1].upper() << "): "
              << h.value(-1) << " +/- " << std::sqrt(h.variance(-1))
              << std::endl;
    std::cout << "overflow  bin [" << h.axis(0_c)[10].lower()
              << ", " << h.axis(0_c)[10].upper() << "): "
              << h.value(10) << " +/- " << std::sqrt(h.variance(10))
              << std::endl;

    /* program output:

    bin 0 x in [-1, -0.7): 1 +/- 1
    bin 1 x in [-0.7, -0.4): 1 +/- 1
    bin 2 x in [-0.4, -0.1): 0 +/- 0
    bin 3 x in [-0.1, 0.2): 2.5 +/- 2.5
    bin 4 x in [0.2, 0.5): 1 +/- 1
    bin 5 x in [0.5, 0.8): 0 +/- 0
    bin 6 x in [0.8, 1.1): 4 +/- 2
    bin 7 x in [1.1, 1.4): 1 +/- 1
    bin 8 x in [1.4, 1.7): 0 +/- 0
    bin 9 x in [1.7, 2): 1 +/- 1
    underflow bin [-inf, -1): 1 +/- 1
    overflow  bin [2, inf): 2 +/- 1.41421

    */
}
