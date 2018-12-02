// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/variable.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/indexed.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <vector>
#include "utility_histogram.hpp"

using namespace boost::histogram;
using namespace boost::histogram::literals;

template <typename Tag>
void run_1d_tests(bool include_extra_bins) {
  auto h = make(Tag(), axis::integer<>(0, 3));
  h(weight(2), 0);
  h(1);
  h(1);

  auto ind = indexed(h, include_extra_bins);
  auto it = ind.begin();
  BOOST_TEST_EQ(it->size(), 1);

  BOOST_TEST_EQ(it->operator[](0), 0);
  BOOST_TEST_EQ(it->value, 2);
  BOOST_TEST_EQ(it->bin(0), h.axis()[0]);
  ++it;
  BOOST_TEST_EQ(it->operator[](0), 1);
  BOOST_TEST_EQ(it->value, 2);
  BOOST_TEST_EQ(it->bin(0), h.axis()[1]);
  ++it;
  BOOST_TEST_EQ(it->operator[](0), 2);
  BOOST_TEST_EQ(it->value, 0);
  BOOST_TEST_EQ(it->bin(0), h.axis()[2]);
  ++it;
  if (include_extra_bins) {
    BOOST_TEST_EQ(it->operator[](0), 3);
    BOOST_TEST_EQ(it->value, 0);
    BOOST_TEST_EQ(it->bin(0), h.axis()[3]);
    ++it;
    BOOST_TEST_EQ(it->operator[](0), -1);
    BOOST_TEST_EQ(it->value, 0);
    BOOST_TEST_EQ(it->bin(0), h.axis()[-1]);
    ++it;
  }
  BOOST_TEST(it == ind.end());
}

template <typename Tag>
void run_2d_tests(bool include_extra_bins) {
  auto h = make_s(Tag(), std::vector<int>(), axis::integer<>(0, 1),
                  axis::integer<int, axis::null_type, axis::option_type::overflow>(2, 4));
  h(weight(2), 0, 2);
  h(-1, 2);
  h(1, 3);

  BOOST_TEST_EQ(axis::traits::extend(h.axis(0)), 3);
  BOOST_TEST_EQ(axis::traits::extend(h.axis(1)), 3);

  auto ind = indexed(h, include_extra_bins);
  auto it = ind.begin();
  BOOST_TEST_EQ(it->size(), 2);

  BOOST_TEST_EQ(it->operator[](0), 0);
  BOOST_TEST_EQ(it->operator[](1), 0);
  BOOST_TEST_EQ(it->bin(0_c), h.axis(0_c)[0]);
  BOOST_TEST_EQ(it->bin(1_c), h.axis(1_c)[0]);
  BOOST_TEST_EQ(it->value, 2);
  ++it;
  if (include_extra_bins) {
    BOOST_TEST_EQ(it->operator[](0), 1);
    BOOST_TEST_EQ(it->operator[](1), 0);
    BOOST_TEST_EQ(it->bin(0), h.axis(0)[1]);
    BOOST_TEST_EQ(it->bin(1), h.axis(1)[0]);
    BOOST_TEST_EQ(it->value, 0);
    ++it;
    BOOST_TEST_EQ(it->operator[](0), -1);
    BOOST_TEST_EQ(it->operator[](1), 0);
    BOOST_TEST_EQ(it->bin(0_c), h.axis(0_c)[-1]);
    BOOST_TEST_EQ(it->bin(1_c), h.axis(1_c)[0]);
    BOOST_TEST_EQ(it->value, 1);
    ++it;
  }
  BOOST_TEST_EQ(it->operator[](0), 0);
  BOOST_TEST_EQ(it->operator[](1), 1);
  BOOST_TEST_EQ(it->bin(0), h.axis(0)[0]);
  BOOST_TEST_EQ(it->bin(1), h.axis(1)[1]);
  BOOST_TEST_EQ(it->value, 0);
  ++it;
  if (include_extra_bins) {
    BOOST_TEST_EQ(it->operator[](0), 1);
    BOOST_TEST_EQ(it->operator[](1), 1);
    BOOST_TEST_EQ(it->bin(0_c), h.axis(0_c)[1]);
    BOOST_TEST_EQ(it->bin(1_c), h.axis(1_c)[1]);
    BOOST_TEST_EQ(it->value, 1);
    ++it;
    BOOST_TEST_EQ(it->operator[](0), -1);
    BOOST_TEST_EQ(it->operator[](1), 1);
    BOOST_TEST_EQ(it->bin(0), h.axis(0)[-1]);
    BOOST_TEST_EQ(it->bin(1), h.axis(1)[1]);
    BOOST_TEST_EQ(it->value, 0);
    ++it;
    BOOST_TEST_EQ(it->operator[](0), 0);
    BOOST_TEST_EQ(it->operator[](1), 2);
    BOOST_TEST_EQ(it->bin(0), h.axis(0)[0]);
    BOOST_TEST_EQ(it->bin(1), h.axis(1)[2]);
    BOOST_TEST_EQ(it->value, 0);
    ++it;
    BOOST_TEST_EQ(it->operator[](0), 1);
    BOOST_TEST_EQ(it->operator[](1), 2);
    BOOST_TEST_EQ(it->bin(0_c), h.axis(0_c)[1]);
    BOOST_TEST_EQ(it->bin(1_c), h.axis(1_c)[2]);
    BOOST_TEST_EQ(it->value, 0);
    ++it;
    BOOST_TEST_EQ(it->operator[](0), -1);
    BOOST_TEST_EQ(it->operator[](1), 2);
    BOOST_TEST_EQ(it->bin(0), h.axis(0)[-1]);
    BOOST_TEST_EQ(it->bin(1), h.axis(1)[2]);
    BOOST_TEST_EQ(it->value, 0);
    ++it;
  }
  BOOST_TEST(it == ind.end());
}

template <typename Tag>
void run_density_tests(bool include_extra_bins) {
  auto ax = axis::variable<>({0.0, 0.1, 0.3, 0.6});
  auto ay = axis::integer<int>(0, 2);
  auto az = ax;
  auto h = make_s(Tag(), std::vector<int>(), ax, ay, az);

  // fill uniformly
  for (unsigned i = 0; i < h.size(); ++i) { unsafe_access::storage(h).set(i, 1); }

  for (auto x : indexed(h, include_extra_bins)) {
    BOOST_TEST_EQ(x.density(), x.value / (x.bin(0).width() * x.bin(2).width()));
  }
}

int main() {
  for (int b = 0; b < 2; ++b) {
    run_1d_tests<static_tag>(b);
    run_1d_tests<dynamic_tag>(b);
    run_2d_tests<static_tag>(b);
    run_2d_tests<dynamic_tag>(b);
    run_density_tests<static_tag>(b);
    run_density_tests<dynamic_tag>(b);
  }
  return boost::report_errors();
}
