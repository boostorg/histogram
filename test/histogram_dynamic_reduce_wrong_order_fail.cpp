// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp>
#include <vector>

using namespace boost::histogram;
int main() {
  auto a = std::vector<axis::variant<axis::integer<>>>();
  a.push_back(axis::integer<>(0, 1));
  a.push_back(axis::integer<>(1, 2));
  auto h = make_histogram(a);
  auto v = {0, 0};
  h.reduce_to(v.begin(), v.end());
}
