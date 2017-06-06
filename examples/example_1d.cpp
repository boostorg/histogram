// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp> // proposed for inclusion in Boost
#include <iostream>
#include <cmath>

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
    h.fill(0.1, bh::weight(5)); // fill with a weighted entry, weight is 5

    // iterate over bins, loop includes under- and overflow bin
    for (const auto& bin : h.axis(0_c)) {
        std::cout << "bin " << bin.idx
                  << " x in [" << bin.left << ", " << bin.right << "): "
                  << h.value(bin.idx) << " +/- " << std::sqrt(h.variance(bin.idx))
                  << std::endl;
    }

    /* program output:

    bin -1 x in [-inf, -1): 1 +/- 1
    bin 0 x in [-1, -0.7): 1 +/- 1
    bin 1 x in [-0.7, -0.4): 1 +/- 1
    bin 2 x in [-0.4, -0.1): 0 +/- 0
    bin 3 x in [-0.1, 0.2): 5 +/- 5
    bin 4 x in [0.2, 0.5): 1 +/- 1
    bin 5 x in [0.5, 0.8): 0 +/- 0
    bin 6 x in [0.8, 1.1): 0 +/- 0
    bin 7 x in [1.1, 1.4): 1 +/- 1
    bin 8 x in [1.4, 1.7): 0 +/- 0
    bin 9 x in [1.7, 2): 1 +/- 1
    bin 10 x in [2, inf): 2 +/- 1.41421

    */
}
