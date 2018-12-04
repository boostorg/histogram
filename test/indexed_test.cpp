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
  BOOST_TEST_EQ(**it, 2);
  BOOST_TEST_EQ(it->bin(0), h.axis()[0]);
  ++it;
  BOOST_TEST_EQ(it->operator[](0), 1);
  BOOST_TEST_EQ(**it, 2);
  BOOST_TEST_EQ(it->bin(0), h.axis()[1]);
  ++it;
  BOOST_TEST_EQ(it->operator[](0), 2);
  BOOST_TEST_EQ(**it, 0);
  BOOST_TEST_EQ(it->bin(0), h.axis()[2]);
  ++it;
  if (include_extra_bins) {
    BOOST_TEST_EQ(it->operator[](0), 3);
    BOOST_TEST_EQ(**it, 0);
    BOOST_TEST_EQ(it->bin(0), h.axis()[3]);
    ++it;
    BOOST_TEST_EQ(it->operator[](0), -1);
    BOOST_TEST_EQ(**it, 0);
    BOOST_TEST_EQ(it->bin(0), h.axis()[-1]);
    ++it;
  }
  BOOST_TEST(it == ind.end());
}

template <typename Tag>
void run_3d_tests(bool b) {
  auto h = make_s(Tag(), std::vector<int>(),
                  axis::integer<>(0, 2),
                  axis::integer<int, axis::null_type, axis::option_type::none>(0, 3),
                  axis::integer<int, axis::null_type, axis::option_type::overflow>(0, 4));

  for (int i = -1; i < 3; ++i)
  for (int j = -1; j < 4; ++j)
  for (int k = -1; k < 5; ++k)
    h(i, j, k, weight(i * 100 + j * 10 + k));

  auto ind = indexed(h, b);
  auto it = ind.begin();
  BOOST_TEST_EQ(it->size(), 3);

  // imitate iteration order of indexed loop
  for (int k = 0; k < 4 + b; ++k)
    for (int j = 0; j < 3; ++j)
      for (int i = 0; i < 2 + 2*b; ++i) {
        const auto i2 = i > 2 ? -1 : i;
        BOOST_TEST_EQ(it->operator[](0), i2);
        BOOST_TEST_EQ(it->operator[](1), j);
        BOOST_TEST_EQ(it->operator[](2), k);
        BOOST_TEST_EQ(it->bin(0_c), h.axis(0_c)[i2]);
        BOOST_TEST_EQ(it->bin(1_c), h.axis(1_c)[j]);
        BOOST_TEST_EQ(it->bin(2_c), h.axis(2_c)[k]);
        BOOST_TEST_EQ(**it, i2 * 100 + j * 10 + k);
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
    BOOST_TEST_EQ(x.density(), *x / (x.bin(0).width() * x.bin(2).width()));
  }
}

int main() {
  for (int b = 0; b < 2; ++b) {
    run_1d_tests<static_tag>(b);
    run_1d_tests<dynamic_tag>(b);
    run_3d_tests<static_tag>(b);
    run_3d_tests<dynamic_tag>(b);
    run_density_tests<static_tag>(b);
    run_density_tests<dynamic_tag>(b);
  }
  return boost::report_errors();
}
