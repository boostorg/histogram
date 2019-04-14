// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include "utility_histogram.hpp"

using namespace boost::histogram;

constexpr auto n_fill = 400000;

template <class Tag, class A1, class A2, class X, class Y>
void fill_test(const A1& a1, const A2& a2, const X& x, const Y& y) {
  auto h1 = make_s(Tag{}, dense_storage<int>(), a1, a2);
  for (unsigned i = 0; i != n_fill; ++i) h1(x[i], y[i]);
  auto h2 = make_s(Tag{}, dense_storage<accumulators::thread_safe<int>>(), a1, a2);
  auto run = [&h2, &x, &y](int k) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    constexpr auto shift = n_fill / 4;
    auto xit = x.cbegin() + k * shift;
    auto yit = y.cbegin() + k * shift;
    for (unsigned i = 0; i < shift; ++i) h2(*xit++, *yit++);
  };

  std::thread t1([&] { run(0); });
  std::thread t2([&] { run(1); });
  std::thread t3([&] { run(2); });
  std::thread t4([&] { run(3); });
  t1.join();
  t2.join();
  t3.join();
  t4.join();

  BOOST_TEST_EQ(algorithm::sum(h1), n_fill);
  BOOST_TEST_EQ(algorithm::sum(h2), n_fill);
  BOOST_TEST_EQ(h1, h2);
}

template <class T>
void tests() {
  std::mt19937 gen(1);
  std::uniform_real_distribution<> rd(-5, 5);
  std::uniform_int_distribution<> id(-5, 5);
  std::vector<double> vd(n_fill);
  std::generate(vd.begin(), vd.end(), [&] { return rd(gen); });
  std::vector<int> vi(n_fill);
  std::generate(vi.begin(), vi.end(), [&] { return id(gen); });

  using r = axis::regular<>;
  using i = axis::integer<>;
  using rg = axis::regular<double, use_default, use_default, axis::option::growth_t>;
  using ig = axis::integer<int, use_default, axis::option::growth_t>;
  fill_test<T>(r(2, 0, 1), i{0, 1}, vd, vi);
  fill_test<T>(rg(2, 0, 1), i{0, 1}, vd, vi);
  fill_test<T>(r(2, 0, 1), ig{0, 1}, vd, vi);
  fill_test<T>(rg(2, 0, 1), ig{0, 1}, vd, vi);
}

int main() {
  tests<static_tag>();
  tests<dynamic_tag>();

  return boost::report_errors();
}