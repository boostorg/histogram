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

using in = axis::integer<int, axis::null_type>;
using in0 = axis::integer<int, axis::null_type, axis::option::none_t>;
using ing = axis::integer<int, axis::null_type,
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
void run_tests() {
  constexpr auto ndata = 1 << 20;

  // 1D simple
  {
    auto h = make(Tag(), in{1, 3});
    auto h3 = h;
    auto h2 = h;

    const int data[1][4] = {{0, 1, 2, 3}};
    for (auto&& x : data[0]) h(x);
    h2.fill(data[0]); // uses 1D specialization
    h3.fill(data);    // uses generic form

    BOOST_TEST_EQ(sum(h), 4);
    BOOST_TEST_EQ(h, h2);
    BOOST_TEST_EQ(h, h3);

    BOOST_TEST_EQ(h2[-1], 1);
    BOOST_TEST_EQ(h2[0], 1);
    BOOST_TEST_EQ(h2[1], 1);
    BOOST_TEST_EQ(h2[2], 1);

    BOOST_TEST_EQ(h3[-1], 1);
    BOOST_TEST_EQ(h3[0], 1);
    BOOST_TEST_EQ(h3[1], 1);
    BOOST_TEST_EQ(h3[2], 1);

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

    const std::array<int, 4> x = {0, 1, 2, 3};
    const std::array<int, 4> y = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i) h(x[i], y[i]);

    const auto xy = {x, y};
    h2.fill(xy);

    BOOST_TEST_EQ(h, h2);

    BOOST_TEST_EQ(h2.at(-1, 0), 1);
    BOOST_TEST_EQ(h2.at(0, 1), 1);
    BOOST_TEST_EQ(h2.at(1, 2), 1);
    BOOST_TEST_EQ(h2.at(2, 3), 1);

    int xy2[2][4] = {{0, 1, 2, 3}, {1, 2, 3, 4}};
    h2.fill(xy2);

    BOOST_TEST_EQ(h2.at(-1, 0), 2);
    BOOST_TEST_EQ(h2.at(0, 1), 2);
    BOOST_TEST_EQ(h2.at(1, 2), 2);
    BOOST_TEST_EQ(h2.at(2, 3), 2);

    BOOST_TEST_THROWS(h2.fill(std::array<std::vector<int>, 2>(
                          {std::vector<int>(2), std::vector<int>(3)})),
                      std::invalid_argument);
  }

  // 1D variant and large input collection and weight
  {
    auto h = make(Tag(), in{1, 3});
    auto h2 = h;

    constexpr auto n = 1 << 20;
    const auto x1 = 1.0;
    std::vector<double> x2(n);
    for (unsigned i = 0; i < n; ++i) x2[i] = i % 4;

    h(x1);
    for (auto&& xi : x2) h(xi);

    using V = variant<double, std::vector<double>>;
    std::vector<V> v(1);
    v[0] = x1;
    h2.fill(v);
    v[0] = x2;
    h2.fill(v);

    BOOST_TEST_EQ(h, h2);

    std::vector<double> w(n);
    for (unsigned i = 0; i < n; ++i) {
      w[i] = i + 1;
      h(weight(w[i]), x2[i]);
    }
    h2.fill(weight(w), v);

    for (unsigned i = 0; i < n; ++i) { h(weight(2), x2[i]); }
    h2.fill(weight(2), v);

    BOOST_TEST_EQ(h, h2);

    w.resize(n - 1);
    BOOST_TEST_THROWS(h2.fill(v, weight(w)), std::invalid_argument);
  }

  // 2D variant and large input collection and weight
  {
    auto h = make(Tag(), in{1, 3}, in0{1, 5});

    std::vector<double> v(ndata);
    for (unsigned i = 0; i < ndata; ++i) v[i] = 1 + i % 4;

    using V = variant<double, std::vector<double>>;
    V xy[2];

    {
      xy[0] = 3;
      xy[1] = v;
      auto h1 = h;
      auto h2 = h;
      for (auto&& vi : v) h1(3, vi);
      h2.fill(xy);
      BOOST_TEST_EQ(h1, h2);
    }

    {
      xy[0] = v;
      xy[1] = 3;
      auto h1 = h;
      auto h2 = h;
      for (auto&& vi : v) h1(vi, 3);
      h2.fill(xy);
      BOOST_TEST_EQ(h1, h2);
    }

    {
      xy[0] = 3;
      xy[1] = v;
      auto h1 = h;
      auto h2 = h;
      for (unsigned i = 0; i < ndata; ++i) { h1(3, v[i], weight(v[i])); }
      h2.fill(xy, weight(v));
      BOOST_TEST_EQ(h1, h2);
    }
  }

  // 1D growing
  {
    auto h = make(Tag(), ing());
    auto h2 = h;
    std::vector<int> x;
    x.reserve(ndata + 2);
    for (unsigned i = 0; i < ndata; ++i) { x.push_back(i % 4); }
    x.push_back(-10);
    x.push_back(10);

    for (auto&& xi : x) h(xi);
    h2.fill(x);

    BOOST_TEST_EQ(h.size(), 23);
    BOOST_TEST_EQ(h2.size(), 23);
    BOOST_TEST_EQ(sum(h), sum(h2));
    BOOST_TEST_EQ(h, h2);
  }

  // 2D growing with weights
  {
    auto h = make(Tag(), in(1, 3), ing());
    auto h2 = h;

    std::vector<int> xy[2];
    for (auto&& v : xy) {
      v.reserve(ndata + 2);
      v.push_back(-10);
      for (unsigned i = 0; i < ndata; ++i) v.push_back(i % 4);
      v.push_back(10);
    }

    for (unsigned i = 0; i < ndata + 2; ++i) h(xy[0][i], xy[1][i]);
    h2.fill(xy);

    BOOST_TEST_EQ(h.size(), 4 * 23);
    BOOST_TEST_EQ(h2.size(), 4 * 23);
    BOOST_TEST_EQ(sum(h), sum(h2));
    BOOST_TEST_EQ(h, h2);
  }

  // 1D profile with samples
  {
    auto h = make_s(Tag(), profile_storage(), in(1, 3));
    auto h2 = h;
    std::vector<double> x;
    x.reserve(ndata);
    for (unsigned i = 0; i < ndata; ++i) { x.push_back(static_cast<double>(i % 4)); }

    for (auto&& xi : x) h(xi, sample(xi));
    h2.fill(x, sample(x));

    BOOST_TEST_EQ(h, h2);

    for (auto&& xi : x) h(xi, sample(xi), weight(xi));
    h2.fill(x, sample(x), weight(x));

    BOOST_TEST_EQ(h, h2);
  }

  // 2D weighted profile with samples and weights
  {
    auto h = make_s(Tag(), weighted_profile_storage(), in(1, 3), in0(1, 3));
    auto h2 = h;
    std::vector<double> x;
    x.reserve(ndata);
    for (unsigned i = 0; i < ndata; ++i) x.push_back(static_cast<double>(i % 4));

    using V = variant<double, std::vector<double>>;
    std::array<V, 2> v;
    v[0] = x;
    v[1] = 3;

    for (auto&& xi : x) h(xi, 3, sample(xi), weight(xi));
    h2.fill(v, sample(x), weight(x));

    BOOST_TEST_EQ(h, h2);
  }

  // axis2d
  {
    auto h = make(Tag(), axis2d{});
    auto h2 = h;

    std::vector<std::tuple<double, double>> xy;
    xy.reserve(ndata);
    for (unsigned i = 0; i < ndata; ++i)
      xy.emplace_back(static_cast<double>(i % 4), static_cast<double>(i % 4));

    for (auto&& xyi : xy) h(xyi);
    h2.fill(xy);

    BOOST_TEST_EQ(h, h2);
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
