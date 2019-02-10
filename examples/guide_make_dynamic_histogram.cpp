// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_make_dynamic_histogram

#include <boost/histogram.hpp>
#include <boost/lexical_cast.hpp> // convert strings to numbers
#include <cassert>
#include <vector>

namespace bh = boost::histogram;

int main(int argc, char** argv) {
  // read axis config from command-line: [nbins start stop] ...
  // and create vector of regular axes, the number of axis is not known at compile-time
  assert(argc - 1 % 3 == 0);
  auto v1 = std::vector<bh::axis::regular<>>();
  for (int iarg = 1; iarg < argc;) {
    const auto bins = boost::lexical_cast<unsigned>(argv[iarg++]);
    const auto start = boost::lexical_cast<double>(argv[iarg++]);
    const auto stop = boost::lexical_cast<double>(argv[iarg++]);
    v1.emplace_back(bins, start, stop);
  }

  // create dynamic histogram from iterator range
  // (copying or moving the vector also works, move is shown below)
  auto h1 = bh::make_histogram(v1.begin(), v1.end());
  assert(h1.rank() == v1.size());

  // create second vector of axis::variant, a polymorphic axis type that can hold concrete
  // axis types, the types and the number of axis can now vary at run-time
  auto v2 = std::vector<bh::axis::variant<bh::axis::regular<>, bh::axis::integer<>>>();
  v2.emplace_back(bh::axis::regular<>(100, -1.0, 1.0));
  v2.emplace_back(bh::axis::integer<>(1, 7));

  // create dynamic histogram by moving the vector
  auto h2 = bh::make_histogram(std::move(v2));
  assert(h2.rank() == 2);
}

//]
