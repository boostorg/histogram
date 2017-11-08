#include <boost/histogram.hpp>
#include <iostream>

int main(int, char**) {
    namespace bh = boost::histogram;
    using namespace bh::literals; // enables _c suffix

    // create 1d-histogram with 10 equidistant bins from -1.0 to 2.0,
    // with axis of histogram labeled as "x"
    auto h = bh::make_static_histogram(bh::axis::regular<>(10, -1.0, 2.0, "x"));

    // fill histogram with data
    h.fill(-1.5); // put in underflow bin
    h.fill(-1.0); // included in first bin, bin interval is semi-open
    h.fill(-0.5);
    h.fill(1.1);
    h.fill(0.3);
    h.fill(1.7);
    h.fill(2.0);  // put in overflow bin, bin interval is semi-open
    h.fill(20.0); // put in overflow bin

    /*
        instead of calling h.fill(...) with same argument N times,
        use bh::count, which accepts an integer argument N
    */
    h.fill(1.0, bh::count(4));

    /*
        to fill a weighted entry, use bh::weight, which accepts a double
        argument; don't confuse with bh::count, it has a different effect
        on the variance (see Rationale for a section explaining weighted fills)
    */
    h.fill(0.1, bh::weight(2.5));

    /*
        iterate over bins, loop excludes under- and overflow bins
        - index 0_c is a compile-time number to make axis(...) return
          a different type for each axis
        - for-loop yields std::pair<[bin index], [bin type]>, where
          [bin type] usually is a semi-open interval representing the bin,
          whose edges can be accessed with methods lower() and upper(), but
          the [bin type] depends on the axis and could be something else
        - value(index) method returns the bin count at index,
        - variance(index) method returns a variance estimate of the bin count
          at index (see Rationale for a section explaining the variance)
    */
    for (const auto& bin : h.axis(0_c)) {
        std::cout << "bin " << bin.first
                  << " x in [" << bin.second.lower() << ", " << bin.second.upper() << "): "
                  << h.value(bin.first) << " +/- " << std::sqrt(h.variance(bin.first))
                  << std::endl;
    }

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

    */
}
