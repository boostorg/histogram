// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <string>
#include <utility>
#include "utility_histogram.hpp"
#include "utility_meta.hpp"

using namespace boost::histogram;

using def = use_default;

using regular = axis::regular<double, def, def, axis::option::growth>;

using integer = axis::integer<
    double, def,
    axis::join<axis::option::underflow, axis::option::overflow, axis::option::growth>>;

using category = axis::category<std::string, def, axis::option::growth>;

class custom_2d_axis {
public:
  auto index(double x, double y) const {
    const auto r = std::sqrt(x * x + y * y);
    return std::min(static_cast<axis::index_type>(r), size());
  }

  auto update(double x, double y) {
    const auto r = std::sqrt(x * x + y * y);
    const auto n = static_cast<int>(r);
    return std::make_pair(n, std::max(size_ - n - 1, 0));
  }

  axis::index_type size() const { return size_; }

private:
  axis::index_type size_ = 0;
};

template <typename Tag>
void run_tests() {
  {
    auto h = make(Tag(), regular(2, 0, 1));
    const auto& a = h.axis();
    BOOST_TEST_EQ(a.size(), 2);
    BOOST_TEST_EQ(h.size(), 2);
    // [0.0, 0.5, 1.0]
    h(0.1);
    h(0.9);
    BOOST_TEST_EQ(a.size(), 2);
    BOOST_TEST_EQ(h.size(), 2);
    h(-std::numeric_limits<double>::infinity());
    h(std::numeric_limits<double>::quiet_NaN());
    h(std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a.size(), 2);
    BOOST_TEST_EQ(h.size(), 2);
    h(-0.3);
    // [-0.5, 0.0, 0.5, 1.0]
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(h.size(), 3);
    BOOST_TEST_EQ(h[0], 1);
    BOOST_TEST_EQ(h[1], 1);
    BOOST_TEST_EQ(h[2], 1);
    h(1.9);
    // [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    BOOST_TEST_EQ(a.size(), 5);
    BOOST_TEST_EQ(h.size(), 5);
    BOOST_TEST_EQ(h[0], 1);
    BOOST_TEST_EQ(h[1], 1);
    BOOST_TEST_EQ(h[2], 1);
    BOOST_TEST_EQ(h[3], 0);
    BOOST_TEST_EQ(h[4], 1);
  }

  {
    auto h = make_s(Tag(), std::vector<int>(), integer());
    const auto& a = h.axis();
    h(-std::numeric_limits<double>::infinity());
    h(std::numeric_limits<double>::quiet_NaN());
    h(std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a.size(), 0);
    BOOST_TEST_EQ(h.size(), 2);
    BOOST_TEST_EQ(h[-1], 1);
    BOOST_TEST_EQ(h[0], 2);
    h(0);
    BOOST_TEST_EQ(a.size(), 1);
    BOOST_TEST_EQ(h.size(), 3);
    BOOST_TEST_EQ(h[-1], 1);
    BOOST_TEST_EQ(h[0], 1);
    BOOST_TEST_EQ(h[1], 2);
    h(2);
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(h.size(), 5);
    BOOST_TEST_EQ(h[-1], 1);
    BOOST_TEST_EQ(h[0], 1);
    BOOST_TEST_EQ(h[1], 0);
    BOOST_TEST_EQ(h[2], 1);
    BOOST_TEST_EQ(h[3], 2);
    h(-2);
    BOOST_TEST_EQ(a.size(), 5);
    BOOST_TEST_EQ(h.size(), 7);
    // BOOST_TEST_EQ(h[-1], 1)
    BOOST_TEST_EQ(h[0], 1);
    BOOST_TEST_EQ(h[1], 0);
    BOOST_TEST_EQ(h[2], 1);
    BOOST_TEST_EQ(h[3], 0);
    BOOST_TEST_EQ(h[4], 1);
    BOOST_TEST_EQ(h[5], 2);
  }

  {
    auto h = make_s(Tag(), std::vector<int>(), integer(), category());
    const auto& a = h.axis(0);
    const auto& b = h.axis(1);
    BOOST_TEST_EQ(a.size(), 0);
    BOOST_TEST_EQ(b.size(), 0);
    BOOST_TEST_EQ(h.size(), 0);
    h(0, "x");
    h(-std::numeric_limits<double>::infinity(), "x");
    h(std::numeric_limits<double>::infinity(), "x");
    h(std::numeric_limits<double>::quiet_NaN(), "x");
    BOOST_TEST_EQ(a.size(), 1);
    BOOST_TEST_EQ(b.size(), 1);
    BOOST_TEST_EQ(h.size(), 3);
    h(2, "x");
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(b.size(), 1);
    BOOST_TEST_EQ(h.size(), 5);
    h(1, "y");
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(b.size(), 2);
    BOOST_TEST_EQ(h.size(), 10);
    BOOST_TEST_EQ(h.at(-1, 0), 1);
    BOOST_TEST_EQ(h.at(-1, 1), 0);
    BOOST_TEST_EQ(h.at(3, 0), 2);
    BOOST_TEST_EQ(h.at(3, 1), 0);
    BOOST_TEST_EQ(h.at(a.index(0), b.index("x")), 1);
    BOOST_TEST_EQ(h.at(a.index(1), b.index("x")), 0);
    BOOST_TEST_EQ(h.at(a.index(2), b.index("x")), 1);
    BOOST_TEST_EQ(h.at(a.index(0), b.index("y")), 0);
    BOOST_TEST_EQ(h.at(a.index(1), b.index("y")), 1);
    BOOST_TEST_EQ(h.at(a.index(2), b.index("y")), 0);
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
