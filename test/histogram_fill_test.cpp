// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/config.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/make_histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/variant2/variant.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include "throw_exception.hpp"
#include "utility_histogram.hpp"

using namespace boost::histogram;
using namespace boost::histogram::algorithm;
using namespace boost::histogram::literals; // to get _c suffix
using boost::variant2::variant;

constexpr auto ndata = 1 << 20;

using in = axis::integer<int, axis::null_type>;
using in0 = axis::integer<int, axis::null_type, axis::option::none_t>;
using ing = axis::integer<double, axis::null_type,
                          decltype(axis::option::growth | axis::option::underflow |
                                   axis::option::overflow)>;

struct axis2d {
  auto size() const { return axis::index_type{2}; }

  auto index(const std::tuple<double, double>& xy) const {
    const auto x = std::get<0>(xy);
    const auto y = std::get<1>(xy);
    const auto r = std::sqrt(x * x + y * y);
    return std::min(static_cast<axis::index_type>(r), size());
  }

  friend std::ostream& operator<<(std::ostream& os, const axis2d&) {
    os << "axis2d()";
    return os;
  }
};

template <class Tag>
void run_tests(const std::vector<int>& x, const std::vector<int>& y) {

  // 1D simple
  {
    auto h = make(Tag(), in{1, 3});
    auto h3 = h;
    auto h2 = h;

    for (auto&& xi : x) h(xi);
    h2.fill(x); // uses 1D specialization
    const auto vx = {x};
    h3.fill(vx); // uses generic form

    BOOST_TEST_EQ(h, h2);
    BOOST_TEST_EQ(h, h3);

#ifndef BOOST_NO_EXCEPTIONS
    int bad1[2][4];
    std::vector<std::array<int, 4>> bad2;
    BOOST_TEST_THROWS(h.fill(bad1), std::invalid_argument);
    BOOST_TEST_THROWS(h.fill(bad2), std::invalid_argument);
#endif
  }

  // 2D simple
  {
    auto h = make(Tag(), in{1, 3}, in0{1, 5});
    auto h2 = h;

    for (int i = 0; i < ndata; ++i) h(x[i], y[i]);
    const auto xy = {x, y};
    h2.fill(xy);

    BOOST_TEST_EQ(h, h2);

    BOOST_TEST_THROWS(h2.fill(std::array<std::vector<int>, 2>(
                          {std::vector<int>(2), std::vector<int>(3)})),
                      std::invalid_argument);
  }

  // 1D variant and weight
  {
    auto h1 = make(Tag(), in{1, 3});
    auto h2 = h1;

    h1(1);
    for (auto&& xi : x) h1(xi);

    using V = variant<double, std::vector<double>>;
    std::vector<V> v(1);
    v[0] = 1;
    h2.fill(v);
    v[0] = std::vector<double>(x.begin(), x.end());
    h2.fill(v);

    BOOST_TEST_EQ(h1, h2);

    for (auto&& xi : x) h1(weight(2), xi);
    h2.fill(weight(2), x);

    BOOST_TEST_EQ(h1, h2);

    std::vector<double> w(y.begin(), y.end());

    for (unsigned i = 0; i < ndata; ++i) h1(weight(w[i]), x[i]);
    h2.fill(weight(w), x);

    BOOST_TEST_EQ(h1, h2);

    w.resize(ndata - 1);
    BOOST_TEST_THROWS(h2.fill(v, weight(w)), std::invalid_argument);
  }

  // 2D variant and weight
  {
    auto h = make(Tag(), in{1, 3}, in0{1, 5});

    using V = variant<int, std::vector<int>>;
    V xy[2];
    std::vector<double> m(x.begin(), x.end());

    {
      xy[0] = 3;
      xy[1] = y;
      auto h1 = h;
      auto h2 = h;
      for (auto&& vi : y) h1(3, vi);
      h2.fill(xy);
      BOOST_TEST_EQ(h1, h2);
    }

    {
      xy[0] = x;
      xy[1] = 3;
      auto h1 = h;
      auto h2 = h;
      for (auto&& vi : x) h1(vi, 3);
      h2.fill(xy);
      BOOST_TEST_EQ(h1, h2);
    }

    {
      xy[0] = 3;
      xy[1] = y;
      auto h1 = h;
      auto h2 = h;
      for (auto&& vi : y) h1(3, vi, weight(2));
      h2.fill(xy, weight(2));
      BOOST_TEST_EQ(h1, h2);
    }

    {
      xy[0] = 3;
      xy[1] = y;
      auto h1 = h;
      auto h2 = h;
      for (unsigned i = 0; i < ndata; ++i) h1(3, y[i], weight(x[i]));
      h2.fill(xy, weight(x));
      BOOST_TEST_EQ(sum(h1), sum(h2));
      BOOST_TEST_EQ(h1, h2);
    }
  }

  // 1D growing
  {
    auto h = make(Tag(), ing());
    auto h2 = h;
    for (const auto& xi : x) h(xi);
    h2.fill(x);
    BOOST_TEST_EQ(h, h2);
  }

  // 2D growing with weights
  {
    auto h = make(Tag(), in(1, 3), ing());
    auto h2 = h;
    for (unsigned i = 0; i < ndata; ++i) h(x[i], y[i]);
    const auto xy = {x, y};
    h2.fill(xy);
    BOOST_TEST_EQ(h, h2);
  }

  {
    auto h = make(Tag(), ing(), ing());
    auto h2 = h;
    for (unsigned i = 0; i < ndata; ++i) h(x[i], y[i]);
    const auto xy = {x, y};
    h2.fill(xy);
    BOOST_TEST_EQ(h, h2);
  }

  // 1D profile with samples
  {
    auto h = make_s(Tag(), profile_storage(), in(1, 3));
    auto h2 = h;
    for (auto&& xi : x) h(xi, sample(static_cast<double>(xi)));
    std::vector<double> m(x.begin(), x.end());
    h2.fill(x, sample(m));
    BOOST_TEST_EQ(h, h2);
    for (auto&& xi : x)
      h(xi, sample(static_cast<double>(xi)), weight(static_cast<double>(xi)));
    h2.fill(x, sample(m), weight(m));
    BOOST_TEST_EQ(h, h2);
  }

  // 2D weighted profile with samples and weights
  {
    auto h = make_s(Tag(), weighted_profile_storage(), in(1, 3), in0(1, 3));
    auto h2 = h;
    std::vector<double> m(x.begin(), x.end());

    using V = variant<double, std::vector<double>>;
    std::array<V, 2> v;
    v[0] = m;
    v[1] = 3;

    for (auto&& vi : m)
      h(vi, 3, sample(static_cast<double>(vi)), weight(static_cast<double>(vi)));
    h2.fill(v, sample(m), weight(m));

    BOOST_TEST_EQ(h, h2);
  }

  // axis2d
  {
    auto h = make(Tag(), axis2d{});
    auto h2 = h;

    std::vector<std::tuple<double, double>> xy;
    xy.reserve(ndata);
    for (unsigned i = 0; i < ndata; ++i) xy.emplace_back(x[i], y[i]);

    for (auto&& xyi : xy) h(xyi);
    h2.fill(xy);

    BOOST_TEST_EQ(h, h2);
  }
}

int main() {
  std::mt19937 gen(1);
  std::uniform_int_distribution<> id(0, 6);
  std::vector<int> x(ndata), y(ndata);
  std::generate(x.begin(), x.end(), [&] { return id(gen); });
  std::generate(y.begin(), y.end(), [&] { return id(gen); });

  run_tests<static_tag>(x, y);
  run_tests<dynamic_tag>(x, y);

  return boost::report_errors();
}
