//[ getting_started_listing_01

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

    // fill histogram with data, typically this happens in a loop
    h(-1.5); // put in underflow bin
    h(-1.0); // included in first bin, bin interval is semi-open
    h(2.0);  // put in overflow bin, bin interval is semi-open
    h(20.0); // put in overflow bin

    // STL algorithms are supported
    auto data = { -0.5, 1.1, 0.3, 1.7 };
    std::for_each(data.begin(), data.end(), h);

    /*
        do a weighted fill using bh::weight, a wrapper for any type,
        which may appear at the beginning of the argument list
    */
    h(bh::weight(1.0), 0.1);

    /*
        iterate over bins, loop excludes under- and overflow bins
        - index 0_c is a compile-time number, the only way in C++ to make
          axis(...) to return a different type for each index
        - for-loop yields instances of `bin_type`, usually is a semi-open
          interval representing the bin, whose edges can be accessed with
          methods `lower()` and `upper()`, but the choice depends on the
          axis type, please look it up in the reference
        - `operator()` returns the bin counter at index, you can then
          access its `value() and `variance()` methods; the first returns the
          actual count, the second returns a variance estimate of the count;
          a bin_type is convertible into an index
          (see Rationale section for what this means)
    */
    for (auto bin : h.axis(0_c)) {
        std::cout << "bin " << bin.idx() << " x in ["
                  << bin.lower() << ", " << bin.upper() << "): "
                  << h.bin(bin).value() << " +/- "
                  << std::sqrt(h.bin(bin).variance())
                  << std::endl;
    }

    // accessing under- and overflow bins is easy, use indices -1 and 10
    std::cout << "underflow bin [" << h.axis(0_c)[-1].lower()
              << ", " << h.axis(0_c)[-1].upper() << "): "
              << h.bin(-1).value() << " +/- " << std::sqrt(h.bin(-1).variance())
              << std::endl;
    std::cout << "overflow  bin [" << h.axis(0_c)[10].lower()
              << ", " << h.axis(0_c)[10].upper() << "): "
              << h.bin(10).value() << " +/- " << std::sqrt(h.bin(10).variance())
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

//]
