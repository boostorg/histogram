// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_custom_storage

#include <array>
#include <unordered_map>
#include <boost/histogram.hpp>
#include <vector>

namespace bh = boost::histogram;

int main() {
  // create static histogram with vector<int> as counter storage,
  // you can use other arithmetic types as counters, e.g. double
  auto h1 = bh::make_histogram_with(std::vector<int>(),
                                    bh::axis::regular<>(10, 0, 1));

  // create static histogram with array<int, N> as counter storage
  // which is allocated completely on the stack (this is very fast)
  auto h2 = bh::make_histogram_with(std::array<int, 12>(),
                                    bh::axis::regular<>(10, 0, 1));
  // N may be larger than the actual number of bins used; an exception
  // is raised if N is too small to hold all bins

  // create static histogram with unordered_map as counter storage;
  // this generates a sparse histogram where only memory is allocated
  // for bins that are non-zero
  auto h3 = bh::make_histogram_with(std::unordered_map<std::size_t, int>(),
                                    bh::axis::regular<>(10, 0, 1));
  // this sounds like a good idea for high-dimensional histograms,
  // but maps come with a signficant memory and run-time overhead; the
  // default_storage usually performs better in high dimensions
}

//]
