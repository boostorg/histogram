// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/ostream_operators.hpp>
#ifndef BOOST_HISTOGRAM_NO_SERIALIZATION
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/histogram/serialization.hpp>
#endif
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include "utility.hpp"

using namespace boost::histogram;
using namespace boost::histogram::literals; // to get _c suffix

int expected_moved_from_dim(static_tag, int static_value) { return static_value; }

int expected_moved_from_dim(dynamic_tag, int) { return 0; }

template <typename A, typename S>
void pass_histogram(boost::histogram::histogram<A, S>& h) {
  BOOST_TEST_EQ(h.at(0), 0);
  BOOST_TEST_EQ(h.at(1), 1);
  BOOST_TEST_EQ(h.at(2), 0);
  BOOST_TEST_EQ(h.axis(0_c), axis::integer<>(0, 3));
}

template <typename Tag>
void run_tests() {
  // init_1
  {
    auto h = make(Tag(), axis::regular<>{3, -1, 1});
    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST_EQ(h.size(), 5);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 5);
    BOOST_TEST_EQ(h.axis().shape(), 5);
    auto h2 = make_s(Tag(), array_storage<unsigned>(), axis::regular<>{3, -1, 1});
    BOOST_TEST_EQ(h2, h);
  }

  // init_2
  {
    auto h = make(Tag(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2});
    BOOST_TEST_EQ(h.dim(), 2);
    BOOST_TEST_EQ(h.size(), 25);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 5);
    BOOST_TEST_EQ(h.axis(1_c).shape(), 5);
    auto h2 = make_s(Tag(), array_storage<unsigned>(), axis::regular<>{3, -1, 1},
                     axis::integer<>{-1, 2});
    BOOST_TEST_EQ(h2, h);
  }

  // init_3
  {
    auto h = make(Tag(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2},
                  axis::circular<>{3});
    BOOST_TEST_EQ(h.dim(), 3);
    BOOST_TEST_EQ(h.size(), 75);
    auto h2 = make_s(Tag(), array_storage<unsigned>(), axis::regular<>{3, -1, 1},
                     axis::integer<>{-1, 2}, axis::circular<>{3});
    BOOST_TEST_EQ(h2, h);
  }

  // init_4
  {
    auto h = make(Tag(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2},
                  axis::circular<>{3}, axis::variable<>{-1, 0, 1});
    BOOST_TEST_EQ(h.dim(), 4);
    BOOST_TEST_EQ(h.size(), 300);
    auto h2 =
        make_s(Tag(), array_storage<unsigned>(), axis::regular<>{3, -1, 1},
               axis::integer<>{-1, 2}, axis::circular<>{3}, axis::variable<>{-1, 0, 1});
    BOOST_TEST_EQ(h2, h);
  }

  // init_5
  {
    enum { A, B, C };
    auto h = make(Tag(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2},
                  axis::circular<>{3}, axis::variable<>{-1, 0, 1},
                  axis::category<>{{A, B, C}});
    BOOST_TEST_EQ(h.dim(), 5);
    BOOST_TEST_EQ(h.size(), 1200);
    auto h2 = make_s(Tag(), array_storage<unsigned>(), axis::regular<>{3, -1, 1},
                     axis::integer<>{-1, 2}, axis::circular<>{3},
                     axis::variable<>{-1, 0, 1}, axis::category<>{{A, B, C}});
    BOOST_TEST_EQ(h2, h);
  }

  // copy_ctor
  {
    auto h = make(Tag(), axis::integer<>{0, 2}, axis::integer<>{0, 3});
    h(0, 0);
    auto h2 = decltype(h)(h);
    BOOST_TEST(h2 == h);
    auto h3 =
        histogram<std::tuple<axis::integer<>, axis::integer<>>, array_storage<unsigned>>(
            h);
    BOOST_TEST_EQ(h3, h);
  }

  // copy_assign
  {
    auto h = make(Tag(), axis::integer<>(0, 1), axis::integer<>(0, 2));
    h(0, 0);
    auto h2 = decltype(h)();
    BOOST_TEST_NE(h, h2);
    h2 = h;
    BOOST_TEST_EQ(h, h2);
    // test self-assign
    h2 = h2;
    BOOST_TEST_EQ(h, h2);
    auto h3 = histogram<std::tuple<axis::integer<>, axis::integer<>>,
                        array_storage<unsigned>>();
    h3 = h;
    BOOST_TEST_EQ(h, h3);
  }

  // move
  {
    auto h = make(Tag(), axis::integer<>(0, 1), axis::integer<>(0, 2));
    h(0, 0);
    const auto href = h;
    decltype(h) h2(std::move(h));
    // static axes cannot shrink to zero
    BOOST_TEST_EQ(h.dim(), expected_moved_from_dim(Tag(), 2));
    BOOST_TEST_EQ(sum(h).value(), 0);
    BOOST_TEST_EQ(h.size(), 0);
    BOOST_TEST_EQ(h2, href);
    decltype(h) h3;
    h3 = std::move(h2);
    // static axes cannot shrink to zero
    BOOST_TEST_EQ(h2.dim(), expected_moved_from_dim(Tag(), 2));
    BOOST_TEST_EQ(sum(h2).value(), 0);
    BOOST_TEST_EQ(h2.size(), 0);
    BOOST_TEST_EQ(h3, href);
  }

  // axis methods
  {
    enum { A = 3, B = 5 };
    auto a = make(Tag(), axis::regular<>(1, 1, 2, "foo"));
    BOOST_TEST_EQ(a.axis().size(), 1);
    BOOST_TEST_EQ(a.axis().shape(), 3);
    BOOST_TEST_EQ(a.axis().index(1), 0);
    BOOST_TEST_EQ(a.axis()[0].lower(), 1);
    BOOST_TEST_EQ(a.axis()[0].upper(), 2);
    BOOST_TEST_EQ(a.axis().label(), "foo");
    a.axis().label("bar");
    BOOST_TEST_EQ(a.axis().label(), "bar");

    auto b = make(Tag(), axis::integer<>(1, 2));
    BOOST_TEST_EQ(b.axis().size(), 1);
    BOOST_TEST_EQ(b.axis().shape(), 3);
    BOOST_TEST_EQ(b.axis().index(1), 0);
    BOOST_TEST_EQ(b.axis()[0].lower(), 1);
    BOOST_TEST_EQ(b.axis()[0].upper(), 2);
    b.axis().label("foo");
    BOOST_TEST_EQ(b.axis().label(), "foo");

    auto c = make(Tag(), axis::category<>({A, B}));
    BOOST_TEST_EQ(c.axis().size(), 2);
    BOOST_TEST_EQ(c.axis().shape(), 3);
    BOOST_TEST_EQ(c.axis().index(A), 0);
    BOOST_TEST_EQ(c.axis().index(B), 1);
    c.axis().label("foo");
    BOOST_TEST_EQ(c.axis().label(), "foo");
    // need to cast here for this to work with Tag == dynamic_tag, too
    auto ca = static_cast<const axis::category<>&>(c.axis());
    BOOST_TEST_EQ(ca[0].value(), A);
  }

  // equal_compare
  {
    auto a = make(Tag(), axis::integer<>(0, 2));
    auto b = make(Tag(), axis::integer<>(0, 2), axis::integer<>(0, 3));
    BOOST_TEST(a != b);
    BOOST_TEST(b != a);
    auto c = make(Tag(), axis::integer<>(0, 2));
    BOOST_TEST(b != c);
    BOOST_TEST(c != b);
    BOOST_TEST(a == c);
    BOOST_TEST(c == a);
    auto d = make(Tag(), axis::regular<>(2, 0, 1));
    BOOST_TEST(c != d);
    BOOST_TEST(d != c);
    c(0);
    BOOST_TEST(a != c);
    BOOST_TEST(c != a);
    a(0);
    BOOST_TEST(a == c);
    BOOST_TEST(c == a);
    a(0);
    BOOST_TEST(a != c);
    BOOST_TEST(c != a);
  }

  // d1
  {
    auto h = make(Tag(), axis::integer<>{0, 2});
    h(0);
    h(0);
    h(-1);
    h(10);

    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST_EQ(h.axis(0_c).size(), 2);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 4);
    BOOST_TEST_EQ(sum(h), 4);

    BOOST_TEST_EQ(h.at(-1), 1);
    BOOST_TEST_EQ(h.at(0), 2);
    BOOST_TEST_EQ(h.at(1), 0);
    BOOST_TEST_EQ(h.at(2), 1);
  }

  // d1_2
  {
    auto h = make(Tag(), axis::integer<>(0, 2, "", axis::uoflow_type::off));
    h(0);
    h(-0);
    h(-1);
    h(10);

    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST_EQ(h.axis(0_c).size(), 2);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 2);
    BOOST_TEST_EQ(sum(h), 2);

    BOOST_TEST_EQ(h.at(0), 2);
    BOOST_TEST_EQ(h.at(1), 0);
  }

  // d1_3
  {
    auto h = make(Tag(), axis::category<std::string>({"A", "B"}));
    h("A");
    h("B");
    h("D");
    h("E");

    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST_EQ(h.axis(0_c).size(), 2);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 3);
    BOOST_TEST_EQ(sum(h), 4);

    BOOST_TEST_EQ(h.at(0), 1);
    BOOST_TEST_EQ(h.at(1), 1);
  }

  // d1w
  {
    auto h = make(Tag(), axis::integer<>(0, 2));
    h(-1);
    h(0);
    h(weight(0.5), 0);
    h(1);
    h(weight(2), 2);

    BOOST_TEST_EQ(sum(h).value(), 5.5);
    BOOST_TEST_EQ(sum(h).variance(), 7.25);

    BOOST_TEST_EQ(h.at(-1).value(), 1);
    BOOST_TEST_EQ(h.at(0).value(), 1.5);
    BOOST_TEST_EQ(h.at(1).value(), 1);
    BOOST_TEST_EQ(h.at(2).value(), 2);

    BOOST_TEST_EQ(h.at(-1).variance(), 1);
    BOOST_TEST_EQ(h.at(0).variance(), 1.25);
    BOOST_TEST_EQ(h.at(1).variance(), 1);
    BOOST_TEST_EQ(h.at(2).variance(), 4);
  }

  // d2
  {
    auto h = make(Tag(), axis::regular<>(2, -1, 1),
                  axis::integer<>(-1, 2, "", axis::uoflow_type::off));
    h(-1, -1);
    h(-1, 0);
    h(-1, -10);
    h(-10, 0);

    BOOST_TEST_EQ(h.dim(), 2);
    BOOST_TEST_EQ(h.axis(0_c).size(), 2);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 4);
    BOOST_TEST_EQ(h.axis(1_c).size(), 3);
    BOOST_TEST_EQ(h.axis(1_c).shape(), 3);
    BOOST_TEST_EQ(sum(h), 3);

    BOOST_TEST_EQ(h.at(-1, 0), 0);
    BOOST_TEST_EQ(h.at(-1, 1), 1);
    BOOST_TEST_EQ(h.at(-1, 2), 0);

    BOOST_TEST_EQ(h.at(0, 0), 1);
    BOOST_TEST_EQ(h.at(0, 1), 1);
    BOOST_TEST_EQ(h.at(0, 2), 0);

    BOOST_TEST_EQ(h.at(1, 0), 0);
    BOOST_TEST_EQ(h.at(1, 1), 0);
    BOOST_TEST_EQ(h.at(1, 2), 0);

    BOOST_TEST_EQ(h.at(2, 0), 0);
    BOOST_TEST_EQ(h.at(2, 1), 0);
    BOOST_TEST_EQ(h.at(2, 2), 0);
  }

  // d2w
  {
    auto h = make(Tag(), axis::regular<>(2, -1, 1),
                  axis::integer<>(-1, 2, "", axis::uoflow_type::off));
    h(-1, 0);              // -> 0, 1
    h(weight(10), -1, -1); // -> 0, 0
    h(weight(5), -1, -10); // is ignored
    h(weight(7), -10, 0);  // -> -1, 1

    BOOST_TEST_EQ(sum(h).value(), 18);
    BOOST_TEST_EQ(sum(h).variance(), 150);

    BOOST_TEST_EQ(h.at(-1, 0).value(), 0);
    BOOST_TEST_EQ(h.at(-1, 1).value(), 7);
    BOOST_TEST_EQ(h.at(-1, 2).value(), 0);

    BOOST_TEST_EQ(h.at(0, 0).value(), 10);
    BOOST_TEST_EQ(h.at(0, 1).value(), 1);
    BOOST_TEST_EQ(h.at(0, 2).value(), 0);

    BOOST_TEST_EQ(h.at(1, 0).value(), 0);
    BOOST_TEST_EQ(h.at(1, 1).value(), 0);
    BOOST_TEST_EQ(h.at(1, 2).value(), 0);

    BOOST_TEST_EQ(h.at(2, 0).value(), 0);
    BOOST_TEST_EQ(h.at(2, 1).value(), 0);
    BOOST_TEST_EQ(h.at(2, 2).value(), 0);

    BOOST_TEST_EQ(h.at(-1, 0).variance(), 0);
    BOOST_TEST_EQ(h.at(-1, 1).variance(), 49);
    BOOST_TEST_EQ(h.at(-1, 2).variance(), 0);

    BOOST_TEST_EQ(h.at(0, 0).variance(), 100);
    BOOST_TEST_EQ(h.at(0, 1).variance(), 1);
    BOOST_TEST_EQ(h.at(0, 2).variance(), 0);

    BOOST_TEST_EQ(h.at(1, 0).variance(), 0);
    BOOST_TEST_EQ(h.at(1, 1).variance(), 0);
    BOOST_TEST_EQ(h.at(1, 2).variance(), 0);

    BOOST_TEST_EQ(h.at(2, 0).variance(), 0);
    BOOST_TEST_EQ(h.at(2, 1).variance(), 0);
    BOOST_TEST_EQ(h.at(2, 2).variance(), 0);
  }

  // d3w
  {
    auto h =
        make(Tag(), axis::integer<>(0, 3), axis::integer<>(0, 4), axis::integer<>(0, 5));
    for (auto i = 0; i < h.axis(0_c).size(); ++i) {
      for (auto j = 0; j < h.axis(1_c).size(); ++j) {
        for (auto k = 0; k < h.axis(2_c).size(); ++k) { h(weight(i + j + k), i, j, k); }
      }
    }

    for (auto i = 0; i < h.axis(0_c).size(); ++i) {
      for (auto j = 0; j < h.axis(1_c).size(); ++j) {
        for (auto k = 0; k < h.axis(2_c).size(); ++k) {
          BOOST_TEST_EQ(h.at(i, j, k).value(), i + j + k);
          BOOST_TEST_EQ(h.at(i, j, k).variance(), (i + j + k) * (i + j + k));
        }
      }
    }
  }

  // add_1
  {
    auto a = make(Tag(), axis::integer<>(0, 2));
    auto b = make_s(Tag(), array_storage<unsigned>(), axis::integer<>(0, 2));
    a(0); // 1 0
    b(1); // 0 1
    auto a2 = a;
    a2 += b;
    BOOST_TEST_EQ(a2.at(-1), 0);
    BOOST_TEST_EQ(a2.at(0), 1);
    BOOST_TEST_EQ(a2.at(1), 1);
    BOOST_TEST_EQ(a2.at(2), 0);
    auto a3 = a;
    a3 += b;
    BOOST_TEST_EQ(a3.at(-1), 0);
    BOOST_TEST_EQ(a3.at(0), 1);
    BOOST_TEST_EQ(a3.at(1), 1);
    BOOST_TEST_EQ(a3.at(2), 0);

    auto c = make(Tag(), axis::integer<>(0, 3));
    BOOST_TEST_THROWS(c += b, std::invalid_argument);
  }

  // add_2
  {
    auto a = make(Tag(), axis::integer<>(0, 2));
    auto b = make(Tag(), axis::integer<>(0, 2));

    a(0);
    BOOST_TEST_EQ(a.at(0).variance(), 1);
    b(weight(3), 1);
    BOOST_TEST_EQ(b.at(1).variance(), 9);
    auto c = a;
    c += b;
    BOOST_TEST_EQ(c.at(-1).value(), 0);
    BOOST_TEST_EQ(c.at(0).value(), 1);
    BOOST_TEST_EQ(c.at(0).variance(), 1);
    BOOST_TEST_EQ(c.at(1).value(), 3);
    BOOST_TEST_EQ(c.at(1).variance(), 9);
    BOOST_TEST_EQ(c.at(2).value(), 0);
    auto d = a;
    d += b;
    BOOST_TEST_EQ(d.at(-1).value(), 0);
    BOOST_TEST_EQ(d.at(0).value(), 1);
    BOOST_TEST_EQ(d.at(0).variance(), 1);
    BOOST_TEST_EQ(d.at(1).value(), 3);
    BOOST_TEST_EQ(d.at(1).variance(), 9);
    BOOST_TEST_EQ(d.at(2).value(), 0);
  }

  // add_3
  {
    auto a = make_s(Tag(), array_storage<char>(), axis::integer<>(-1, 2));
    auto b = make_s(Tag(), array_storage<unsigned>(), axis::integer<>(-1, 2));
    a(-1);
    b(1);
    auto c = a;
    c += b;
    BOOST_TEST_EQ(c.at(-1), 0);
    BOOST_TEST_EQ(c.at(0), 1);
    BOOST_TEST_EQ(c.at(1), 0);
    BOOST_TEST_EQ(c.at(2), 1);
    BOOST_TEST_EQ(c.at(3), 0);
    auto d = a;
    d += b;
    BOOST_TEST_EQ(d.at(-1), 0);
    BOOST_TEST_EQ(d.at(0), 1);
    BOOST_TEST_EQ(d.at(1), 0);
    BOOST_TEST_EQ(d.at(2), 1);
    BOOST_TEST_EQ(d.at(3), 0);
  }

  // functional programming
  {
    auto v = std::vector<int>{0, 1, 2};
    auto h = std::for_each(v.begin(), v.end(), make(Tag(), axis::integer<>(0, 3)));
    BOOST_TEST_EQ(h.at(0), 1);
    BOOST_TEST_EQ(h.at(1), 1);
    BOOST_TEST_EQ(h.at(2), 1);
    BOOST_TEST_EQ(sum(h), 3);
  }

  // operators
  {
    auto a = make(Tag(), axis::integer<>(0, 3));
    auto b = a;
    a(0);
    b(1);
    auto c = a + b;
    BOOST_TEST_EQ(c.at(0), 1);
    BOOST_TEST_EQ(c.at(1), 1);
    c += b;
    BOOST_TEST_EQ(c.at(0), 1);
    BOOST_TEST_EQ(c.at(1), 2);
    auto d = a + b + c;
    BOOST_TEST_EQ(d.at(0), 2);
    BOOST_TEST_EQ(d.at(1), 3);
    auto e = 3 * a;
    auto f = b * 2;
    BOOST_TEST_EQ(e.at(0).value(), 3);
    BOOST_TEST_EQ(e.at(1).value(), 0);
    BOOST_TEST_EQ(f.at(0).value(), 0);
    BOOST_TEST_EQ(f.at(1).value(), 2);
    auto r = a;
    r += b;
    r += e;
    BOOST_TEST_EQ(r.at(0).value(), 4);
    BOOST_TEST_EQ(r.at(1).value(), 1);
    BOOST_TEST_EQ(r, a + b + 3 * a);
    auto s = r / 4;
    r /= 4;
    BOOST_TEST_EQ(r.at(0).value(), 1);
    BOOST_TEST_EQ(r.at(1).value(), 0.25);
    BOOST_TEST_EQ(r, s);
  }

#ifndef BOOST_HISTOGRAM_NO_SERIALIZATION
  // histogram_serialization
  {
    enum { A, B, C };
    auto a = make(
        Tag(), axis::regular<>(3, -1, 1, "r"), axis::circular<>(4, 0.0, 1.0, "p"),
        axis::regular<axis::transform::log>(3, 1, 100, "lr"),
        axis::regular<axis::transform::pow>(3, 1, 100, "pr", axis::uoflow_type::on, 0.5),
        axis::variable<>({0.1, 0.2, 0.3, 0.4, 0.5}, "v"), axis::category<>{A, B, C},
        axis::integer<>(0, 2, "i"));
    a(0.5, 0.2, 20, 20, 0.25, 1, 1);
    std::string buf;
    {
      std::ostringstream os;
      boost::archive::text_oarchive oa(os);
      oa << a;
      buf = os.str();
    }
    auto b = decltype(a)();
    BOOST_TEST_NE(a, b);
    {
      std::istringstream is(buf);
      boost::archive::text_iarchive ia(is);
      ia >> b;
    }
    BOOST_TEST_EQ(a, b);
  }
#endif

  // histogram_ostream
  {
    auto a = make(Tag(), axis::regular<>(3, -1, 1, "r"), axis::integer<>(0, 2, "i"));
    std::ostringstream os;
    os << a;
    BOOST_TEST_EQ(os.str(),
                  "histogram("
                  "\n  regular(3, -1, 1, label='r'),"
                  "\n  integer(0, 2, label='i'),"
                  "\n)");
  }

  // histogram_reset
  {
    auto h = make(Tag(), axis::integer<>(0, 2, "", axis::uoflow_type::off));
    h(0);
    h(1);
    BOOST_TEST_EQ(h.at(0), 1);
    BOOST_TEST_EQ(h.at(1), 1);
    BOOST_TEST_EQ(sum(h), 2);
    h.reset();
    BOOST_TEST_EQ(h.at(0), 0);
    BOOST_TEST_EQ(h.at(1), 0);
    BOOST_TEST_EQ(sum(h), 0);
  }

  // reduce
  {
    auto h1 = make(Tag(), axis::integer<>(0, 2), axis::integer<>(0, 3));
    h1(0, 0);
    h1(0, 1);
    h1(1, 0);
    h1(1, 1);
    h1(1, 2);

    /*
      matrix layout:

      0 ->
    1 1 1 0 0
    | 1 1 0 0
    v 0 1 0 0
      0 0 0 0
      0 0 0 0
    */

    auto h1_0 = h1.reduce_to(0_c);
    BOOST_TEST_EQ(h1_0.dim(), 1);
    BOOST_TEST_EQ(sum(h1_0), 5);
    BOOST_TEST_EQ(h1_0.at(0), 2);
    BOOST_TEST_EQ(h1_0.at(1), 3);
    BOOST_TEST_EQ(h1_0.axis()[0].lower(), 0);
    BOOST_TEST_EQ(h1_0.axis()[1].lower(), 1);
    BOOST_TEST(h1_0.axis() == h1.axis(0_c));

    auto h1_1 = h1.reduce_to(1_c);
    BOOST_TEST_EQ(h1_1.dim(), 1);
    BOOST_TEST_EQ(sum(h1_1), 5);
    BOOST_TEST_EQ(h1_1.at(0), 2);
    BOOST_TEST_EQ(h1_1.at(1), 2);
    BOOST_TEST_EQ(h1_1.at(2), 1);
    BOOST_TEST(h1_1.axis() == h1.axis(1_c));

    auto h2 =
        make(Tag(), axis::integer<>(0, 2), axis::integer<>(0, 3), axis::integer<>(0, 4));
    h2(0, 0, 0);
    h2(0, 1, 0);
    h2(0, 1, 1);
    h2(0, 0, 2);
    h2(1, 0, 2);

    auto h2_0 = h2.reduce_to(0_c);
    BOOST_TEST_EQ(h2_0.dim(), 1);
    BOOST_TEST_EQ(sum(h2_0), 5);
    BOOST_TEST_EQ(h2_0.at(0), 4);
    BOOST_TEST_EQ(h2_0.at(1), 1);
    BOOST_TEST(h2_0.axis() == axis::integer<>(0, 2));

    auto h2_1 = h2.reduce_to(1_c);
    BOOST_TEST_EQ(h2_1.dim(), 1);
    BOOST_TEST_EQ(sum(h2_1), 5);
    BOOST_TEST_EQ(h2_1.at(0), 3);
    BOOST_TEST_EQ(h2_1.at(1), 2);
    BOOST_TEST(h2_1.axis() == axis::integer<>(0, 3));

    auto h2_2 = h2.reduce_to(2_c);
    BOOST_TEST_EQ(h2_2.dim(), 1);
    BOOST_TEST_EQ(sum(h2_2), 5);
    BOOST_TEST_EQ(h2_2.at(0), 2);
    BOOST_TEST_EQ(h2_2.at(1), 1);
    BOOST_TEST_EQ(h2_2.at(2), 2);
    BOOST_TEST(h2_2.axis() == axis::integer<>(0, 4));

    auto h2_01 = h2.reduce_to(0_c, 1_c);
    BOOST_TEST_EQ(h2_01.dim(), 2);
    BOOST_TEST_EQ(sum(h2_01), 5);
    BOOST_TEST_EQ(h2_01.at(0, 0), 2);
    BOOST_TEST_EQ(h2_01.at(0, 1), 2);
    BOOST_TEST_EQ(h2_01.at(1, 0), 1);
    BOOST_TEST(h2_01.axis(0_c) == axis::integer<>(0, 2));
    BOOST_TEST(h2_01.axis(1_c) == axis::integer<>(0, 3));

    auto h2_02 = h2.reduce_to(0_c, 2_c);
    BOOST_TEST_EQ(h2_02.dim(), 2);
    BOOST_TEST_EQ(sum(h2_02), 5);
    BOOST_TEST_EQ(h2_02.at(0, 0), 2);
    BOOST_TEST_EQ(h2_02.at(0, 1), 1);
    BOOST_TEST_EQ(h2_02.at(0, 2), 1);
    BOOST_TEST_EQ(h2_02.at(1, 2), 1);
    BOOST_TEST(h2_02.axis(0_c) == axis::integer<>(0, 2));
    BOOST_TEST(h2_02.axis(1_c) == axis::integer<>(0, 4));

    auto h2_12 = h2.reduce_to(1_c, 2_c);
    BOOST_TEST_EQ(h2_12.dim(), 2);
    BOOST_TEST_EQ(sum(h2_12), 5);
    BOOST_TEST_EQ(h2_12.at(0, 0), 1);
    BOOST_TEST_EQ(h2_12.at(1, 0), 1);
    BOOST_TEST_EQ(h2_12.at(1, 1), 1);
    BOOST_TEST_EQ(h2_12.at(0, 2), 2);
    BOOST_TEST(h2_12.axis(0_c) == axis::integer<>(0, 3));
    BOOST_TEST(h2_12.axis(1_c) == axis::integer<>(0, 4));
  }

  // custom axis
  {
    struct custom_axis : public axis::integer<> {
      using value_type = const char*; // type that is fed to the axis

      using integer::integer; // inherit ctors of base

      // the customization point
      // - accept const char* and convert to int
      // - then call index method of base class
      int index(value_type s) const { return integer::index(std::atoi(s)); }
    };

    auto h = make(Tag(), custom_axis(0, 3));
    h("-10");
    h("0");
    h("1");
    h("9");

    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST_EQ(h.axis(), custom_axis(0, 3));
    BOOST_TEST_EQ(h.at(0), 1);
    BOOST_TEST_EQ(h.at(1), 1);
    BOOST_TEST_EQ(h.at(2), 0);
  }

  // histogram iterator 1D
  {
    auto h = make(Tag(), axis::integer<>(0, 3));
    const auto& a = h.axis();
    h(weight(2), 0);
    h(1);
    h(1);
    auto it = h.begin();
    BOOST_TEST_EQ(it.dim(), 1);

    BOOST_TEST_EQ(it.idx(), 0);
    BOOST_TEST_EQ(it.bin(), a[0]);
    BOOST_TEST_EQ(it->value(), 2);
    BOOST_TEST_EQ(it->variance(), 4);
    ++it;
    BOOST_TEST_EQ(it.idx(), 1);
    BOOST_TEST_EQ(it.bin(), a[1]);
    BOOST_TEST_EQ(*it, 2);
    ++it;
    BOOST_TEST_EQ(it.idx(), 2);
    BOOST_TEST_EQ(it.bin(), a[2]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST_EQ(it.idx(), 3);
    BOOST_TEST_EQ(it.bin(), a[3]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST_EQ(it.idx(), -1);
    BOOST_TEST_EQ(it.bin(), a[-1]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST(it == h.end());
  }

  // histogram iterator 2D
  {
    auto h = make(Tag(), axis::integer<>(0, 1),
                  axis::integer<>(2, 4, "", axis::uoflow_type::off));
    const auto& a0 = h.axis(0_c);
    const auto& a1 = h.axis(1_c);
    h(weight(2), 0, 2);
    h(-1, 2);
    h(1, 3);

    auto it = h.begin();
    BOOST_TEST_EQ(it.dim(), 2);

    BOOST_TEST_EQ(it.idx(0), 0);
    BOOST_TEST_EQ(it.idx(1), 0);
    BOOST_TEST_EQ(it.bin(0_c), a0[0]);
    BOOST_TEST_EQ(it.bin(1_c), a1[0]);
    BOOST_TEST_EQ(it->value(), 2);
    BOOST_TEST_EQ(it->variance(), 4);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 1);
    BOOST_TEST_EQ(it.idx(1), 0);
    BOOST_TEST_EQ(it.bin(0_c), a0[1]);
    BOOST_TEST_EQ(it.bin(1_c), a1[0]);
    BOOST_TEST_EQ(it->value(), 0);
    BOOST_TEST_EQ(it->variance(), 0);
    ++it;
    BOOST_TEST_EQ(it.idx(0), -1);
    BOOST_TEST_EQ(it.idx(1), 0);
    BOOST_TEST_EQ(it.bin(0_c), a0[-1]);
    BOOST_TEST_EQ(it.bin(1_c), a1[0]);
    BOOST_TEST_EQ(it->value(), 1);
    BOOST_TEST_EQ(it->variance(), 1);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 0);
    BOOST_TEST_EQ(it.idx(1), 1);
    BOOST_TEST_EQ(it.bin(0_c), a0[0]);
    BOOST_TEST_EQ(it.bin(1_c), a1[1]);
    BOOST_TEST_EQ(it->value(), 0);
    BOOST_TEST_EQ(it->variance(), 0);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 1);
    BOOST_TEST_EQ(it.idx(1), 1);
    BOOST_TEST_EQ(it.bin(0_c), a0[1]);
    BOOST_TEST_EQ(it.bin(1_c), a1[1]);
    BOOST_TEST_EQ(it->value(), 1);
    BOOST_TEST_EQ(it->variance(), 1);
    ++it;
    BOOST_TEST_EQ(it.idx(0), -1);
    BOOST_TEST_EQ(it.idx(1), 1);
    BOOST_TEST_EQ(it.bin(0_c), a0[-1]);
    BOOST_TEST_EQ(it.bin(1_c), a1[1]);
    BOOST_TEST_EQ(it->value(), 0);
    BOOST_TEST_EQ(it->variance(), 0);
    ++it;
    BOOST_TEST(it == h.end());

    auto v = sum(h);
    BOOST_TEST_EQ(v.value(), 4);
    BOOST_TEST_EQ(v.variance(), 6);
  }

  // STL compatibility
  {
    auto h = make(Tag(), axis::integer<>(0, 3));
    for (int i = 0; i < 3; ++i) h(i);
    auto a = std::vector<weight_counter<double>>();
    std::partial_sum(h.begin(), h.end(), std::back_inserter(a));
    BOOST_TEST_EQ(a[0].value(), 1);
    BOOST_TEST_EQ(a[1].value(), 2);
    BOOST_TEST_EQ(a[2].value(), 3);
  }

  // using STL containers
  {
    auto h = make(Tag(), axis::integer<>(0, 2), axis::regular<>(2, 2, 4));
    // vector in
    h(std::vector<int>({0, 2}));
    // pair in
    h(std::make_pair(1, 3.0));

    // pair out
    BOOST_TEST_EQ(h.at(std::make_pair(0, 0)), 1);
    BOOST_TEST_EQ(h[std::make_pair(0, 0)], 1);
    // tuple out
    BOOST_TEST_EQ(h[std::make_tuple(1, 1)], 1);

    // vector in, weights
    h(weight(2), std::vector<int>({0, 2}));
    // pair in, weights
    h(weight(2), std::make_pair(1, 3.0));

    // vector
    BOOST_TEST_EQ(h.at(std::vector<int>({0, 0})).value(), 3);
    BOOST_TEST_EQ(h[std::vector<int>({0, 0})].value(), 3);
    // initializer_list
    auto i = {1, 1};
    BOOST_TEST_EQ(h.at(i).variance(), 5);
    BOOST_TEST_EQ(h[i].variance(), 5);
  }

  // bad bin access
  {
    auto h = make(Tag(), axis::integer<>(0, 1), axis::integer<>(0, 1));
    BOOST_TEST_THROWS(h.at(0, 2), std::out_of_range);
    BOOST_TEST_THROWS(h.at(std::make_pair(2, 0)), std::out_of_range);
    BOOST_TEST_THROWS(h.at(std::vector<int>({0, 2})), std::out_of_range);
  }

  // pass histogram to function
  {
    auto h = make(Tag(), axis::integer<>(0, 3));
    h(1);
    pass_histogram(h);
  }

  // allocator support
  {
    tracing_allocator_db db;
    {
      tracing_allocator<char> a(db);
      auto h = make_s(Tag(), array_storage<int, tracing_allocator<int>>(a),
                      axis::integer<int, tracing_allocator<char>>(
                          0, 1024, std::string(512, 'c'), axis::uoflow_type::on, a));
      h(0);
    }

    // int allocation for array_storage
    BOOST_TEST_EQ(db[typeid(int)].first, db[typeid(int)].second);
    BOOST_TEST_GE(db[typeid(int)].first, 1024u);

    // char allocation for axis label
    BOOST_TEST_EQ(db[typeid(char)].first, db[typeid(char)].second);
    BOOST_TEST_GE(db[typeid(char)].first, 512u);

    if (Tag()) { // axis::any allocation, only for dynamic histogram
      using T = axis::any<axis::integer<int, tracing_allocator<char>>>;
      BOOST_TEST_EQ(db[typeid(T)].first, db[typeid(T)].second);
      BOOST_TEST_GE(db[typeid(T)].first, 1u);
    }
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
