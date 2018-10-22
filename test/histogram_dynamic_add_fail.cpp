// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp>
#include <vector>

using namespace boost::histogram;
int main() {
  auto va = std::vector<axis::variant<axis::integer<>>>();
  va.push_back(axis::integer<>(0, 2));
  auto a = make_histogram(va);

  auto vb = std::vector<axis::variant<axis::integer<>>>();
  vb.push_back(axis::integer<>(0, 3));
  auto b = make_histogram(vb);

  a += b;
}
