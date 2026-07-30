// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/detail/detect.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/throw_exception.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <vector>
#include "allocator.hpp"
#include "dummy_storage.hpp"
#include "histogram.hpp"
#include "ostream.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram;

template <typename Tag>
void run_tests() {
  // arithmetic operators
  {
    auto a = make(Tag(), axis::integer<>(0, 2));
    auto b = a;
    a.at(-1) = 2;
    a.at(0) = 1;
    b.at(-1) = 1;
    b.at(1) = 1;
    b.at(2) = 3;

    auto c = a + b;
    BOOST_TEST_EQ(c.at(-1), 3);
    BOOST_TEST_EQ(c.at(0), 1);
    BOOST_TEST_EQ(c.at(1), 1);
    BOOST_TEST_EQ(c.at(2), 3);

    c += b;
    BOOST_TEST_EQ(c.at(-1), 4);
    BOOST_TEST_EQ(c.at(0), 1);
    BOOST_TEST_EQ(c.at(1), 2);
    BOOST_TEST_EQ(c.at(2), 6);

    auto d = a + b + c;
    BOOST_TEST_TRAIT_SAME(decltype(d), decltype(a));
    BOOST_TEST_EQ(d.at(-1), 7);
    BOOST_TEST_EQ(d.at(0), 2);
    BOOST_TEST_EQ(d.at(1), 3);
    BOOST_TEST_EQ(d.at(2), 9);

    auto d2 = d - a - b - c;
    BOOST_TEST_TRAIT_SAME(decltype(d2), decltype(a));
    BOOST_TEST_EQ(d2.at(-1), 0);
    BOOST_TEST_EQ(d2.at(0), 0);
    BOOST_TEST_EQ(d2.at(1), 0);
    BOOST_TEST_EQ(d2.at(2), 0);

    d2 -= a;
    BOOST_TEST_EQ(d2.at(-1), -2);
    BOOST_TEST_EQ(d2.at(0), -1);
    BOOST_TEST_EQ(d2.at(1), 0);
    BOOST_TEST_EQ(d2.at(2), 0);

    auto d3 = d;
    d3 *= d;
    BOOST_TEST_EQ(d3.at(-1), 49);
    BOOST_TEST_EQ(d3.at(0), 4);
    BOOST_TEST_EQ(d3.at(1), 9);
    BOOST_TEST_EQ(d3.at(2), 81);

    auto d4 = d3 * (1 * d); // converted return type
    BOOST_TEST_TRAIT_FALSE((std::is_same<decltype(d4), decltype(d3)>));
    BOOST_TEST_EQ(d4.at(0), 8);
    BOOST_TEST_EQ(d4.at(1), 27);
    d4 /= d;
    BOOST_TEST_EQ(d4.at(0), 4);
    BOOST_TEST_EQ(d4.at(1), 9);
    auto d5 = d4 / d;
    BOOST_TEST_EQ(d5.at(0), 2);
    BOOST_TEST_EQ(d5.at(1), 3);

    auto e = 3 * a; // converted return type
    auto f = b * 2; // converted return type
    BOOST_TEST_TRAIT_FALSE((std::is_same<decltype(e), decltype(a)>));
    BOOST_TEST_TRAIT_FALSE((std::is_same<decltype(f), decltype(a)>));
    BOOST_TEST_EQ(e.at(0), 3);
    BOOST_TEST_EQ(e.at(1), 0);
    BOOST_TEST_EQ(f.at(0), 0);
    BOOST_TEST_EQ(f.at(1), 2);
    auto r = 1.0 * a;
    r += b;
    r += e;
    BOOST_TEST_EQ(r.at(0), 4);
    BOOST_TEST_EQ(r.at(1), 1);
    BOOST_TEST_EQ(r, a + b + 3 * a);
    auto s = r / 4;
    r /= 4;
    BOOST_TEST_EQ(r.at(0), 1);
    BOOST_TEST_EQ(r.at(1), 0.25);
    BOOST_TEST_EQ(r, s);
  }

  // arithmetic operators with mixed storage: unlimited vs. vector<unsigned>
  {
    auto ia = axis::integer<int, axis::null_type, axis::option::none_t>(0, 2);
    auto a = make(Tag(), ia);
    a(0, weight(2));
    a(1, weight(2));
    auto b = a;
    auto c = make_s(Tag(), std::vector<int>(), ia);
    c(0, weight(2));
    c(1, weight(2));
    auto a2 = a;
    a2 += c;
    BOOST_TEST_EQ(a2, (a + b));
    auto a3 = a;
    a3 *= c;
    BOOST_TEST_EQ(a3, (a * b));
    auto a4 = a;
    a4 -= c;
    BOOST_TEST_EQ(a4, (a - b));
    auto a5 = a;
    a5 /= c;
    BOOST_TEST_EQ(a5, (a / b));
  }

  // arithmetic operators with mixed storage: vector<unsigned char> vs. vector<unsigned>
  {
    auto ia = axis::integer<int, axis::null_type, axis::option::none_t>(0, 2);
    auto a = make_s(Tag(), std::vector<unsigned long>{}, ia);
    auto c = make_s(Tag(), std::vector<unsigned>(), ia);
    a(0, weight(2u));
    a(1, weight(2u));
    auto b = a;
    c(0, weight(2u));
    c(1, weight(2u));
    auto a2 = a;
    a2 += c;
    BOOST_TEST_EQ(a2, (a + b));
    auto a3 = a;
    a3 *= c;
    BOOST_TEST_EQ(a3, (a * b));
    auto a4 = a;
    a4 -= c;
    BOOST_TEST_EQ(a4, (a - b));
    auto a5 = a;
    a5 /= c;
    BOOST_TEST_EQ(a5, (a / b));
  }

  // add operators with weighted storage
  {
    auto ia = axis::integer<int, axis::null_type, axis::option::none_t>(0, 2);
    auto a = make_s(Tag(), std::vector<accumulators::weighted_sum<>>(), ia);
    auto b = make_s(Tag(), std::vector<accumulators::weighted_sum<>>(), ia);

    a(0);
    b(weight(3), 1);
    BOOST_TEST_EQ(a.at(0).variance(), 1);
    BOOST_TEST_EQ(b.at(1).variance(), 9);
    auto c = a;
    c += b;
    BOOST_TEST_EQ(c.at(0).value(), 1);
    BOOST_TEST_EQ(c.at(0).variance(), 1);
    BOOST_TEST_EQ(c.at(1).value(), 3);
    BOOST_TEST_EQ(c.at(1).variance(), 9);
    auto d = a;
    d += b;
    BOOST_TEST_EQ(d.at(0).value(), 1);
    BOOST_TEST_EQ(d.at(0).variance(), 1);
    BOOST_TEST_EQ(d.at(1).value(), 3);
    BOOST_TEST_EQ(d.at(1).variance(), 9);

    // add unweighted histogram
    auto e = make_s(Tag(), std::vector<int>(), ia);
    std::fill(e.begin(), e.end(), 2);

    d += e;
    BOOST_TEST_EQ(d.at(0).value(), 3);
    BOOST_TEST_EQ(d.at(0).variance(), 3);
    BOOST_TEST_EQ(d.at(1).value(), 5);
    BOOST_TEST_EQ(d.at(1).variance(), 11);
  }

  // division operators with weighted storage
  {
    using w_t = accumulators::weighted_sum<>;
    auto ia = axis::integer<int, axis::null_type, axis::option::none_t>(0, 2);
    auto a = make_s(Tag(), std::vector<w_t>(), ia);
    auto b = make_s(Tag(), std::vector<w_t>(), ia);

    a(0);
    a(1, weight(2));
    b(weight(4), 0);
    b(weight(3), 1);

    w_t av[2] = {w_t{1, 1}, w_t{2, 4}};
    w_t bv[2] = {w_t{4, 16}, w_t{3, 9}};
    BOOST_TEST_EQ(a.at(0), av[0]);
    BOOST_TEST_EQ(a.at(1), av[1]);
    BOOST_TEST_EQ(b.at(0), bv[0]);
    BOOST_TEST_EQ(b.at(1), bv[1]);

    auto c = a;
    c /= b;

    w_t cv[2] = {av[0], av[1]};
    cv[0] /= bv[0];
    cv[1] /= bv[1];
    BOOST_TEST_EQ(c.at(0), cv[0]);
    BOOST_TEST_EQ(c.at(1), cv[1]);

    // division by unweighted histogram
    auto e = make_s(Tag(), std::vector<double>(), ia);

    e(0);
    e(0);
    e(1);

    auto f = a / e;
    w_t fv[2] = {av[0], av[1]};
    fv[0] /= e.at(0);
    fv[1] /= e.at(1);

    BOOST_TEST_EQ(f.at(0), fv[0]);
    BOOST_TEST_EQ(f.at(1), fv[1]);
  }

  // merging add
  {
    using C = axis::category<int, use_default, axis::option::growth_t>;
    using I = axis::integer<int, axis::null_type, axis::option::growth_t>;

    {
      auto empty = std::initializer_list<int>{};
      auto a = make(Tag(), C(empty, "foo"));
      auto b = make(Tag(), C(empty, "foo"));
      a(2);
      a(1);
      b(2);
      b(3);
      b(4);
      a += b;
      BOOST_TEST_EQ(a.axis(), C({2, 1, 3, 4}, "foo"));
      BOOST_TEST_EQ(a[0], 2);
      BOOST_TEST_EQ(a[1], 1);
      BOOST_TEST_EQ(a[2], 1);
      BOOST_TEST_EQ(a[3], 1);
    }

    {
      auto a = make(Tag(), C{1, 2}, I{4, 5});
      auto b = make(Tag(), C{2, 3}, I{5, 6});

      std::fill(a.begin(), a.end(), 1);
      std::fill(b.begin(), b.end(), 1);

      a += b;

      BOOST_TEST_EQ(a.axis(0), (C{1, 2, 3}));
      BOOST_TEST_EQ(a.axis(1), (I{4, 6}));
      BOOST_TEST_EQ(a.at(0, 0), 1);
      BOOST_TEST_EQ(a.at(1, 0), 1);
      BOOST_TEST_EQ(a.at(2, 0), 0); // v=(3, 4) did not exist in a or b
      BOOST_TEST_EQ(a.at(0, 1), 0); // v=(1, 5) did not exist in a or b
      BOOST_TEST_EQ(a.at(1, 1), 1);
      BOOST_TEST_EQ(a.at(2, 1), 1);
    }

    {
      using CI = axis::category<int, use_default, axis::option::growth_t>;
      using CS = axis::category<std::string, use_default, axis::option::growth_t>;

      auto h1 = make(Tag{}, CI{}, CS{});
      auto h2 = make(Tag{}, CI{}, CS{});
      auto h3 = make(Tag{}, CI{}, CS{});

      h1(1, "b");
      h1(2, "a");
      h1(1, "a");
      h1(2, "b");

      h2(2, "b");
      h2(3, "b");
      h2(4, "c");
      h2(5, "c");

      h3(1, "b");
      h3(2, "a");
      h3(1, "a");
      h3(2, "b");
      h3(2, "b");
      h3(3, "b");
      h3(4, "c");
      h3(5, "c");

      BOOST_TEST_EQ(h3, h1 + h2);
    }

    {
      // C2 is not growing and has overflow
      using C2 = axis::category<int>;
      auto a = make(Tag(), C{1, 2}, C2{4, 5});
      auto b = make(Tag(), C{1, 2}, C2{5, 6});

      // axis C2 is incompatible
      BOOST_TEST_THROWS(a += b, std::invalid_argument);

      std::fill(a.begin(), a.end(), 1);
      b = a;
      b(3, 4);
      a += b;
      BOOST_TEST_EQ(a.at(0, 0), 2);
      BOOST_TEST_EQ(a.at(1, 0), 2);
      BOOST_TEST_EQ(a.at(2, 0), 1);
      BOOST_TEST_EQ(a.at(0, 1), 2);
      BOOST_TEST_EQ(a.at(1, 1), 2);
      BOOST_TEST_EQ(a.at(2, 1), 0);
      BOOST_TEST_EQ(a.at(0, 2), 2);
      BOOST_TEST_EQ(a.at(1, 2), 2);
      BOOST_TEST_EQ(a.at(2, 2), 0);

      // incompatible labels
      b.axis(0).metadata() = "foo";
      BOOST_TEST_THROWS(a += b, std::invalid_argument);

      // incompatible axis types
      auto c = make(Tag(), C{1, 2}, I{4, 6});
      BOOST_TEST_THROWS(a += c, std::invalid_argument);
    }
  }

  // arithmetic with sparse storage: map-backed histograms must not densify
  {
    using map_t = std::map<std::size_t, double, std::less<std::size_t>,
                           tracing_allocator<std::pair<const std::size_t, double>>>;
    tracing_allocator_db db;
    tracing_allocator<char> alloc(db);
    auto ia = axis::integer<int, axis::null_type, axis::option::none_t>(0, 100);

    auto a = make_s(Tag(), map_t(alloc), ia);
    auto b = make_s(Tag(), map_t(alloc), ia);
    const auto baseline = db.first;
    a(5); // causes one allocation
    const auto node = db.first - baseline;
    BOOST_TEST_GT(node, 0);
    b(7); // causes one allocation
    BOOST_TEST_EQ(db.first, baseline + 2 * node);

    // dense references, all checks below compare against dense results
    auto ra = make_s(Tag(), std::vector<double>(), ia);
    auto rb = make_s(Tag(), std::vector<double>(), ia);
    ra(5);
    rb(7);

    // map += map allocates one node, not one per cell
    auto t0 = db.first;
    a += b;
    BOOST_TEST_EQ(db.first, t0 + node);
    auto rab = ra;
    rab += rb;
    BOOST_TEST_EQ(a, rab);

    // map -= map allocates one node holding a negative value
    auto c = make_s(Tag(), map_t(alloc), ia);
    t0 = db.first;
    c -= b;
    BOOST_TEST_EQ(db.first, t0 + node);
    auto rc = make_s(Tag(), std::vector<double>(), ia);
    rc -= rb;
    BOOST_TEST_EQ(c, rc);

    // self-addition doubles values without new allocations
    auto d = a;
    t0 = db.first;
    d += d;
    BOOST_TEST_EQ(db.first, t0);
    auto rd = rab;
    rd += rd;
    BOOST_TEST_EQ(d, rd);

    // map += dense allocates only for non-zero cells of rhs
    auto e = make_s(Tag(), map_t(alloc), ia);
    e(3);
    t0 = db.first;
    e += ra;
    BOOST_TEST_EQ(db.first, t0 + node);
    auto re = make_s(Tag(), std::vector<double>(), ia);
    re(3);
    re += ra;
    BOOST_TEST_EQ(e, re);

    // dense += map
    auto f = make_s(Tag(), std::vector<double>(), ia);
    f(2);
    f += b;
    auto rf = make_s(Tag(), std::vector<double>(), ia);
    rf(2);
    rf += rb;
    BOOST_TEST_EQ(f, rf);

    // default storage += map
    auto g = make(Tag(), ia);
    g(2);
    g += b;
    BOOST_TEST_EQ(g, rf);

    // mixed value types: map of double += vector of int
    auto h = make_s(Tag(), map_t(alloc), ia);
    auto vi = make_s(Tag(), std::vector<int>(), ia);
    vi(4);
    t0 = db.first;
    h += vi;
    BOOST_TEST_EQ(db.first, t0 + node);
    BOOST_TEST_EQ(h.at(4), 1);

    // map *= map does not allocate
    auto m = a; // a has entries at 5 and 7
    t0 = db.first;
    m *= b;
    BOOST_TEST_EQ(db.first, t0);
    BOOST_TEST_EQ(m.at(5), 0);
    BOOST_TEST_EQ(m.at(7), 1);

    // map /= map keeps dense semantics: 0/0 is NaN, 1/0 is inf
    auto n = a;
    n /= b;
    BOOST_TEST_EQ(n.at(7), 1);
    BOOST_TEST_EQ(n.at(5), std::numeric_limits<double>::infinity());
    BOOST_TEST(std::isnan(static_cast<double>(n.at(0))));

    // merging add with growing axes and map storage on both sides
    using C = axis::category<int, use_default, axis::option::growth_t>;
    auto p = make_s(Tag(), map_t(alloc), C{1, 2});
    auto q = make_s(Tag(), map_t(alloc), C{2, 3});
    p(1);
    p(2);
    q(2);
    q(3);
    q(3);
    p += q;
    auto rp = make_s(Tag(), std::vector<double>(), C{1, 2});
    auto rq = make_s(Tag(), std::vector<double>(), C{2, 3});
    rp(1);
    rp(2);
    rq(2);
    rq(3);
    rq(3);
    rp += rq;
    BOOST_TEST_EQ(p, rp);
  }

  // bad operations
  {
    auto a = make(Tag(), axis::regular<>(2, 0, 4));
    auto b = make(Tag(), axis::regular<>(2, 0, 2));
    BOOST_TEST_THROWS(a += b, std::invalid_argument);
    BOOST_TEST_THROWS(a -= b, std::invalid_argument);
    BOOST_TEST_THROWS(a *= b, std::invalid_argument);
    BOOST_TEST_THROWS(a /= b, std::invalid_argument);
    auto c = make(Tag(), axis::regular<>(2, 0, 2), axis::regular<>(2, 0, 4));
    BOOST_TEST_THROWS(a += c, std::invalid_argument);
  }

  // scaling
  {
    auto b = make_s(Tag{}, dummy_storage<double, true>{}, axis::integer<>(0, 1));
    b(0);
    BOOST_TEST_EQ(b[0], 1);
    b *= 2; // intentionally does not do anything
    BOOST_TEST_EQ(b[0], 1);

    auto c = make_s(Tag{}, dummy_storage<double, false>{}, axis::integer<>(0, 1));
    c(0);
    BOOST_TEST_EQ(c[0], 1);
    c *= 2; // this calls *= on each element
    BOOST_TEST_EQ(c[0], 2);

    using h1_t = decltype(make_s(Tag{}, dummy_storage<unscaleable, false>{},
                                 axis::integer<>(0, 1)));
    BOOST_TEST_NOT((detail::has_operator_rmul<h1_t, double>::value));

    using h2_t = decltype(make_s(Tag{}, dummy_storage<unscaleable, true>{},
                                 axis::integer<>(0, 1)));
    BOOST_TEST_NOT((detail::has_operator_rmul<h2_t, double>::value));
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  // histograms with different axes types and map-backed rhs
  {
    auto ia = axis::integer<int, axis::null_type, axis::option::none_t>(0, 3);
    auto a = make_s(static_tag(), std::vector<double>(), ia);
    auto b = make_s(dynamic_tag(), std::map<std::size_t, double>(), ia);
    a(0);
    b(1);
    a += b;
    BOOST_TEST_EQ(a.at(0), 1);
    BOOST_TEST_EQ(a.at(1), 1);
    a -= b;
    BOOST_TEST_EQ(a.at(0), 1);
    BOOST_TEST_EQ(a.at(1), 0);
  }

  return boost::report_errors();
}
