// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/option.hpp>
#include <iostream>

using namespace boost::histogram::axis;

template <unsigned N, unsigned M>
bool operator==(option_set<N>, option_set<M>) {
  return N == M;
}

template <unsigned N>
std::ostream& operator<<(std::ostream& os, option_set<N>) {
  os << "underflow " << bool(N & option::underflow::value) << " "
     << "overflow " << bool(N & option::overflow::value) << " "
     << "circular " << bool(N & option::circular::value) << " "
     << "growth " << bool(N & option::growth::value);
  return os;
}

int main() {
  using namespace option;
  using uoflow = join<underflow, overflow>;
  constexpr auto uoflow_growth = join<uoflow, growth>{};
  constexpr auto overflow_growth = join<overflow, growth>{};
  constexpr auto overflow_circular = join<overflow, circular>{};

  BOOST_TEST_EQ(uoflow::value, underflow::value | overflow::value);
  BOOST_TEST((test<uoflow, underflow>::value));
  BOOST_TEST((test<uoflow, overflow>::value));
  BOOST_TEST_NOT((test<uoflow, circular>::value));
  BOOST_TEST_NOT((test<uoflow, growth>::value));
  BOOST_TEST((test<underflow, underflow>::value));

  BOOST_TEST_EQ((join<growth, circular>{}), circular{});
  BOOST_TEST_EQ((join<circular, underflow>{}), underflow{});
  BOOST_TEST_EQ((join<underflow, underflow, overflow>{}), uoflow{});
  BOOST_TEST_EQ((join<circular, overflow, underflow, growth>{}), uoflow_growth);
  BOOST_TEST_EQ((join<uoflow, circular, growth>{}), overflow_growth);
  BOOST_TEST_EQ((join<growth, overflow, underflow, circular>{}), overflow_circular);
  BOOST_TEST_EQ((join<uoflow, growth, circular>{}), overflow_circular);
  BOOST_TEST_EQ((join<growth, circular, overflow, underflow>{}), uoflow{});
  return boost::report_errors();
}
