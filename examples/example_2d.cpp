// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cstdlib>

namespace br = boost::random;
namespace bh = boost::histogram;

int main() {
    using namespace bh::literals; // for _c
    br::mt19937 gen;
    br::normal_distribution<> norm;
    auto h = bh::make_static_histogram(
        bh::axis::regular<>(5, -5, 5, "x"),
        bh::axis::regular<>(5, -5, 5, "y")
    );

    // fill histogram
    for (int i = 0; i < 1000; ++i)
        h.fill(norm(gen), norm(gen));

    // show histogram
    for (const auto& ybin : h.axis(1_c)) { // vertical
        for (const auto& xbin : h.axis(0_c)) { // horizontal
            std::printf("%3.0f ", h.value(xbin.first, ybin.first));
        }
        std::printf("\n");
    }
}
