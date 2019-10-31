// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/make_profile.hpp>

int main() {
  using namespace boost::histogram;
  auto h = make_profile(axis::integer<>(0, 5));
  h(0, sample(1, 2)); // weighted profile requires one sample
}
