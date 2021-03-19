// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/accumulators/thread_safe.hpp>
#include <sstream>
#include <thread>
#include "throw_exception.hpp"
#include "utility_str.hpp"

using namespace boost::histogram;
using namespace std::literals;

template <class F>
void parallel(F f) {
  auto g = [&]() {
    for (int i = 0; i < 1000; ++i) f();
  };

  std::thread a(g), b(g), c(g), d(g);
  a.join();
  b.join();
  c.join();
  d.join();
}

template <class T>
void test_on() {
  using ts_t = accumulators::thread_safe<T>;

  // default ctor
  {
    ts_t i;
    BOOST_TEST_EQ(i, static_cast<T>(0));
  }

  // ctor from value
  {
    ts_t i{1001};
    BOOST_TEST_EQ(i, 1001);
    BOOST_TEST_EQ(str(i), "1001"s);
  }

  // add null
  BOOST_TEST_EQ(ts_t{} += ts_t{}, ts_t{});

  // add non-null
  BOOST_TEST_EQ((ts_t{} += ts_t{2}), (ts_t{2}));

  // operator++
  {
    ts_t t;
    parallel([&]() { ++t; });
    BOOST_TEST_EQ(t, 4000);
  }

  // operator+= with value
  {
    ts_t t;
    parallel([&]() { t += 2; });
    BOOST_TEST_EQ(t, 8000);
  }

  // operator+= with another thread_safe
  {
    ts_t t, u;
    u = 2;
    parallel([&]() { t += u; });
    BOOST_TEST_EQ(t, 8000);
  }
}

int main() {
  test_on<int>();
  test_on<float>();

  return boost::report_errors();
}
