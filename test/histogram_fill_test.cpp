// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
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

constexpr auto ndata = 1 << 16; // should be larger than index buffer in fill_n

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
void run_tests(const std::vector<int>& x, const std::vector<int>& y,
               const std::vector<double>& w) {

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

#ifndef BOOST_NO_EXCEPTIONS
    // wrong rank
    BOOST_TEST_THROWS(h.fill(x), std::invalid_argument);

    // not rectangular
    std::array<std::vector<int>, 2> bad = {{std::vector<int>(2), std::vector<int>(3)}};
    BOOST_TEST_THROWS(h2.fill(bad), std::invalid_argument);
#endif
  }

  // 1D variant and weight
  {
    auto h1 = make(Tag(), in{1, 3});
    auto h2 = h1;

    h1(1);
    for (auto&& xi : x) h1(xi);

    using V = variant<int, std::vector<int>>;
    std::vector<V> v(1);
    v[0] = 1;
    h2.fill(v);
    v[0] = x;
    h2.fill(v);

    BOOST_TEST_EQ(h1, h2);

    for (auto&& xi : x) h1(weight(2), xi);
    h2.fill(weight(2), x);

    BOOST_TEST_EQ(h1, h2);

    for (unsigned i = 0; i < ndata; ++i) h1(weight(w[i]), x[i]);
    h2.fill(weight(w), x);

    BOOST_TEST_EQ(h1, h2);

#ifndef BOOST_NO_EXCEPTIONS
    auto w2 = w;
    w2.resize(ndata - 1);
    BOOST_TEST_THROWS(h2.fill(v, weight(w2)), std::invalid_argument);
#endif
  }

  // 2D variant and weight
  {
    auto h = make(Tag(), in{1, 3}, in0{1, 5});

    using V = variant<int, std::vector<int>>;
    V xy[2];

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
      for (unsigned i = 0; i < ndata; ++i) h1(3, y[i], weight(w[i]));
      h2.fill(xy, weight(w));
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

  // 2D growing A
  {
    auto h = make(Tag(), in(1, 3), ing());
    auto h2 = h;
    for (unsigned i = 0; i < ndata; ++i) h(x[i], y[i]);
    const auto xy = {x, y};
    h2.fill(xy);
    BOOST_TEST_EQ(h, h2);
  }

  // 2D growing B
  {
    auto h = make(Tag(), ing(), ing());
    auto h2 = h;
    for (unsigned i = 0; i < ndata; ++i) h(x[i], y[i]);
    const auto xy = {x, y};
    h2.fill(xy);
    BOOST_TEST_EQ(h, h2);
  }

  // 2D growing A with weights
  {
    auto h = make(Tag(), in(1, 3), ing());
    auto h2 = h;
    for (unsigned i = 0; i < ndata; ++i) h(x[i], y[i], weight(w[i]));
    const auto xy = {x, y};
    h2.fill(xy, weight(w));
    BOOST_TEST_EQ(h, h2);
  }

  // 2D growing B with weights
  {
    auto h = make(Tag(), ing(), ing());
    auto h2 = h;
    for (unsigned i = 0; i < ndata; ++i) h(x[i], y[i], weight(w[i]));
    const auto xy = {x, y};
    h2.fill(xy, weight(w));
    BOOST_TEST_EQ(h, h2);
  }

  // 1D profile with samples
  {
    auto h = make_s(Tag(), profile_storage(), in(1, 3));
    auto h2 = h;
    for (unsigned i = 0; i < ndata; ++i) h(x[i], sample(w[i]));

    h2.fill(x, sample(w));
    BOOST_TEST_EQ(h, h2);
    for (unsigned i = 0; i < ndata; ++i) h(x[i], sample(w[i]), weight(w[i]));
    h2.fill(x, sample(w), weight(w));
    BOOST_TEST_EQ(h, h2);
  }

  // 2D weighted profile with samples and weights
  {
    auto h = make_s(Tag(), weighted_profile_storage(), in(1, 3), in0(1, 3));
    auto h2 = h;

    using V = variant<int, std::vector<int>>;
    std::array<V, 2> xy;
    xy[0] = x;
    xy[1] = 3;

    for (unsigned i = 0; i < ndata; ++i) h(x[i], 3, sample(w[i]), weight(w[i]));
    h2.fill(xy, sample(w), weight(w));

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
  std::normal_distribution<> id(0, 2);
  std::vector<int> x(ndata), y(ndata);
  auto generator = [&] { return static_cast<int>(id(gen)); };
  std::generate(x.begin(), x.end(), generator);
  std::generate(y.begin(), y.end(), generator);
  std::vector<double> w(ndata);
  // must be all positive
  std::generate(w.begin(), w.end(), [&] { return 0.5 + std::abs(id(gen)); });

  run_tests<static_tag>(x, y, w);
  run_tests<dynamic_tag>(x, y, w);

  return boost::report_errors();
}
