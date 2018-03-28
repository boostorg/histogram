// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/static_histogram.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/mpl/int.hpp>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

using namespace boost::histogram;
using namespace boost::histogram::literals; // to get _c suffix
namespace mpl = boost::mpl;

template <typename S, typename... Axes>
auto make_histogram(static_tag, Axes &&... axes)
    -> decltype(make_static_histogram_with<S>(std::forward<Axes>(axes)...)) {
  return make_static_histogram_with<S>(std::forward<Axes>(axes)...);
}

template <typename S, typename... Axes>
auto make_histogram(dynamic_tag, Axes &&... axes)
    -> decltype(make_dynamic_histogram_with<S>(std::forward<Axes>(axes)...)) {
  return make_dynamic_histogram_with<S>(std::forward<Axes>(axes)...);
}

template <typename T, typename U>
bool axis_equal(static_tag, const T &t, const U &u) {
  return t == u;
}

template <typename T, typename U>
bool axis_equal(dynamic_tag, const T &t, const U &u) {
  return t == T(u); // need to convert rhs to boost::variant
}

int expected_moved_from_dim(static_tag, int static_value) {
  return static_value;
}

int expected_moved_from_dim(dynamic_tag, int) { return 0; }

template <typename... Ts>
void pass_histogram(boost::histogram::histogram<Ts...> &h) {}

template <typename Type> void run_tests() {

  // init_1
  {
    auto h =
        make_histogram<adaptive_storage>(Type(), axis::regular<>{3, -1, 1});
    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST_EQ(h.bincount(), 5);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 5);
    BOOST_TEST_EQ(h.axis().shape(), 5);
    auto h2 = make_histogram<array_storage<unsigned>>(
        Type(), axis::regular<>{3, -1, 1});
    BOOST_TEST_EQ(h2, h);
  }

  // init_2
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::regular<>{3, -1, 1},
                                              axis::integer<>{-1, 2});
    BOOST_TEST_EQ(h.dim(), 2);
    BOOST_TEST_EQ(h.bincount(), 25);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 5);
    BOOST_TEST_EQ(h.axis(1_c).shape(), 5);
    auto h2 = make_histogram<array_storage<unsigned>>(
        Type(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2});
    BOOST_TEST_EQ(h2, h);
  }

  // init_3
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::regular<>{3, -1, 1},
                                              axis::integer<>{-1, 2},
                                              axis::circular<>{3});
    BOOST_TEST_EQ(h.dim(), 3);
    BOOST_TEST_EQ(h.bincount(), 75);
    auto h2 = make_histogram<array_storage<unsigned>>(
        Type(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2},
        axis::circular<>{3});
    BOOST_TEST_EQ(h2, h);
  }

  // init_4
  {
    auto h = make_histogram<adaptive_storage>(
        Type(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2},
        axis::circular<>{3}, axis::variable<>{-1, 0, 1});
    BOOST_TEST_EQ(h.dim(), 4);
    BOOST_TEST_EQ(h.bincount(), 300);
    auto h2 = make_histogram<array_storage<unsigned>>(
        Type(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2},
        axis::circular<>{3}, axis::variable<>{-1, 0, 1});
    BOOST_TEST_EQ(h2, h);
  }

  // init_5
  {
    enum { A, B, C };
    auto h = make_histogram<adaptive_storage>(
        Type(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2},
        axis::circular<>{3}, axis::variable<>{-1, 0, 1},
        axis::category<>{{A, B, C}});
    BOOST_TEST_EQ(h.dim(), 5);
    BOOST_TEST_EQ(h.bincount(), 900);
    auto h2 = make_histogram<array_storage<unsigned>>(
        Type(), axis::regular<>{3, -1, 1}, axis::integer<>{-1, 2},
        axis::circular<>{3}, axis::variable<>{-1, 0, 1},
        axis::category<>{{A, B, C}});
    BOOST_TEST_EQ(h2, h);
  }

  // copy_ctor
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>{0, 2},
                                              axis::integer<>{0, 3});
    h.fill(0, 0);
    auto h2 = decltype(h)(h);
    BOOST_TEST(h2 == h);
    auto h3 = static_histogram<mpl::vector<axis::integer<>, axis::integer<>>,
                               array_storage<unsigned>>(h);
    BOOST_TEST_EQ(h3, h);
  }

  // copy_assign
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 1),
                                              axis::integer<>(0, 2));
    h.fill(0, 0);
    auto h2 = decltype(h)();
    BOOST_TEST_NE(h, h2);
    h2 = h;
    BOOST_TEST_EQ(h, h2);
    // test self-assign
    h2 = h2;
    BOOST_TEST_EQ(h, h2);
    auto h3 = static_histogram<mpl::vector<axis::integer<>, axis::integer<>>,
                               array_storage<unsigned>>();
    h3 = h;
    BOOST_TEST_EQ(h, h3);
  }

  // move
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 1),
                                              axis::integer<>(0, 2));
    h.fill(0, 0);
    const auto href = h;
    decltype(h) h2(std::move(h));
    // static axes cannot shrink to zero
    BOOST_TEST_EQ(h.dim(), expected_moved_from_dim(Type(), 2));
    BOOST_TEST_EQ(h.sum().value(), 0);
    BOOST_TEST_EQ(h.bincount(), 0);
    BOOST_TEST_EQ(h2, href);
    decltype(h) h3;
    h3 = std::move(h2);
    // static axes cannot shrink to zero
    BOOST_TEST_EQ(h2.dim(), expected_moved_from_dim(Type(), 2));
    BOOST_TEST_EQ(h2.sum().value(), 0);
    BOOST_TEST_EQ(h2.bincount(), 0);
    BOOST_TEST_EQ(h3, href);
  }

  // axis methods
  {
    enum { A = 3, B = 5 };
    auto a = make_histogram<adaptive_storage>(Type(),
                                              axis::regular<>(1, 1, 2, "foo"));
    BOOST_TEST_EQ(a.axis().size(), 1);
    BOOST_TEST_EQ(a.axis().shape(), 3);
    BOOST_TEST_EQ(a.axis().index(1), 0);
    BOOST_TEST_EQ(a.axis()[0].lower(), 1);
    BOOST_TEST_EQ(a.axis()[0].upper(), 2);
    BOOST_TEST_EQ(a.axis().label(), "foo");
    a.axis().label("bar");
    BOOST_TEST_EQ(a.axis().label(), "bar");

    auto b = make_histogram<adaptive_storage>(Type(), axis::integer<>(1, 2));
    BOOST_TEST_EQ(b.axis().size(), 1);
    BOOST_TEST_EQ(b.axis().shape(), 3);
    BOOST_TEST_EQ(b.axis().index(1), 0);
    BOOST_TEST_EQ(b.axis()[0].lower(), 1);
    BOOST_TEST_EQ(b.axis()[0].upper(), 2);
    b.axis().label("foo");
    BOOST_TEST_EQ(b.axis().label(), "foo");

    auto c = make_histogram<adaptive_storage>(Type(), axis::category<>({A, B}));
    BOOST_TEST_EQ(c.axis().size(), 2);
    BOOST_TEST_EQ(c.axis().shape(), 2);
    BOOST_TEST_EQ(c.axis().index(A), 0);
    BOOST_TEST_EQ(c.axis().index(B), 1);
    c.axis().label("foo");
    BOOST_TEST_EQ(c.axis().label(), "foo");
    // need to cast here for this to work with Type == dynamic_tag
    auto ca = axis::cast<axis::category<>>(c.axis());
    BOOST_TEST_EQ(ca[0].value(), A);
  }

  // equal_compare
  {
    auto a = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2));
    auto b = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2),
                                              axis::integer<>(0, 3));
    BOOST_TEST(a != b);
    BOOST_TEST(b != a);
    auto c = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2));
    BOOST_TEST(b != c);
    BOOST_TEST(c != b);
    BOOST_TEST(a == c);
    BOOST_TEST(c == a);
    auto d = make_histogram<adaptive_storage>(Type(), axis::regular<>(2, 0, 1));
    BOOST_TEST(c != d);
    BOOST_TEST(d != c);
    c.fill(0);
    BOOST_TEST(a != c);
    BOOST_TEST(c != a);
    a.fill(0);
    BOOST_TEST(a == c);
    BOOST_TEST(c == a);
    a.fill(0);
    BOOST_TEST(a != c);
    BOOST_TEST(c != a);
  }

  // d1
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>{0, 2});
    h.fill(0);
    h.fill(0);
    h.fill(-1);
    h.fill(10);

    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST_EQ(h.axis(0_c).size(), 2);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 4);
    BOOST_TEST_EQ(h.sum(), 4);

    BOOST_TEST_THROWS(h(-2), std::out_of_range);
    BOOST_TEST_EQ(h(-1), 1);
    BOOST_TEST_EQ(h(0), 2);
    BOOST_TEST_EQ(h(1), 0);
    BOOST_TEST_EQ(h(2), 1);
    BOOST_TEST_THROWS(h(3), std::out_of_range);
  }

  // d1_2
  {
    auto h = make_histogram<adaptive_storage>(
        Type(), axis::integer<>(0, 2, "", axis::uoflow::off));
    h.fill(0);
    h.fill(-0);
    h.fill(-1);
    h.fill(10);

    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST_EQ(h.axis(0_c).size(), 2);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 2);
    BOOST_TEST_EQ(h.sum(), 2);

    BOOST_TEST_THROWS(h(-1), std::out_of_range);
    BOOST_TEST_EQ(h(0), 2);
    BOOST_TEST_EQ(h(1), 0);
    BOOST_TEST_THROWS(h(2), std::out_of_range);
  }

  // d1_3
  {
    auto h = make_histogram<adaptive_storage>(
        Type(), axis::category<std::string>({"A", "B"}));
    h.fill("A");
    h.fill("B");
    h.fill("D");
    h.fill("E");

    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST_EQ(h.axis(0_c).size(), 2);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 2);
    BOOST_TEST_EQ(h.sum(), 2);

    BOOST_TEST_THROWS(h(-1), std::out_of_range);
    BOOST_TEST_EQ(h(0), 1);
    BOOST_TEST_EQ(h(1), 1);
    BOOST_TEST_THROWS(h(2), std::out_of_range);
  }

  // d1w
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2));
    h.fill(-1);
    h.fill(0);
    h.fill(weight(0.5), 0);
    h.fill(1);
    h.fill(2, weight(2));

    BOOST_TEST_EQ(h.sum().value(), 5.5);
    BOOST_TEST_EQ(h.sum().variance(), 7.25);

    BOOST_TEST_EQ(h(-1).value(), 1);
    BOOST_TEST_EQ(h(0).value(), 1.5);
    BOOST_TEST_EQ(h(1).value(), 1);
    BOOST_TEST_EQ(h(2).value(), 2);

    BOOST_TEST_EQ(h(-1).variance(), 1);
    BOOST_TEST_EQ(h(0).variance(), 1.25);
    BOOST_TEST_EQ(h(1).variance(), 1);
    BOOST_TEST_EQ(h(2).variance(), 4);
  }

  // d2
  {
    auto h = make_histogram<adaptive_storage>(
        Type(), axis::regular<>(2, -1, 1),
        axis::integer<>(-1, 2, "", axis::uoflow::off));
    h.fill(-1, -1);
    h.fill(-1, 0);
    h.fill(-1, -10);
    h.fill(-10, 0);

    BOOST_TEST_EQ(h.dim(), 2);
    BOOST_TEST_EQ(h.axis(0_c).size(), 2);
    BOOST_TEST_EQ(h.axis(0_c).shape(), 4);
    BOOST_TEST_EQ(h.axis(1_c).size(), 3);
    BOOST_TEST_EQ(h.axis(1_c).shape(), 3);
    BOOST_TEST_EQ(h.sum(), 3);

    BOOST_TEST_EQ(h(-1, 0), 0);
    BOOST_TEST_EQ(h(-1, 1), 1);
    BOOST_TEST_EQ(h(-1, 2), 0);

    BOOST_TEST_EQ(h(0, 0), 1);
    BOOST_TEST_EQ(h(0, 1), 1);
    BOOST_TEST_EQ(h(0, 2), 0);

    BOOST_TEST_EQ(h(1, 0), 0);
    BOOST_TEST_EQ(h(1, 1), 0);
    BOOST_TEST_EQ(h(1, 2), 0);

    BOOST_TEST_EQ(h(2, 0), 0);
    BOOST_TEST_EQ(h(2, 1), 0);
    BOOST_TEST_EQ(h(2, 2), 0);
  }

  // d2w
  {
    auto h = make_histogram<adaptive_storage>(
        Type(), axis::regular<>(2, -1, 1),
        axis::integer<>(-1, 2, "", axis::uoflow::off));
    h.fill(-1, 0);              // -> 0, 1
    h.fill(weight(10), -1, -1); // -> 0, 0
    h.fill(weight(5), -1, -10); // is ignored
    h.fill(weight(7), -10, 0);  // -> -1, 1

    BOOST_TEST_EQ(h.sum().value(), 18);
    BOOST_TEST_EQ(h.sum().variance(), 150);

    BOOST_TEST_EQ(h(-1, 0).value(), 0);
    BOOST_TEST_EQ(h(-1, 1).value(), 7);
    BOOST_TEST_EQ(h(-1, 2).value(), 0);

    BOOST_TEST_EQ(h(0, 0).value(), 10);
    BOOST_TEST_EQ(h(0, 1).value(), 1);
    BOOST_TEST_EQ(h(0, 2).value(), 0);

    BOOST_TEST_EQ(h(1, 0).value(), 0);
    BOOST_TEST_EQ(h(1, 1).value(), 0);
    BOOST_TEST_EQ(h(1, 2).value(), 0);

    BOOST_TEST_EQ(h(2, 0).value(), 0);
    BOOST_TEST_EQ(h(2, 1).value(), 0);
    BOOST_TEST_EQ(h(2, 2).value(), 0);

    BOOST_TEST_EQ(h(-1, 0).variance(), 0);
    BOOST_TEST_EQ(h(-1, 1).variance(), 49);
    BOOST_TEST_EQ(h(-1, 2).variance(), 0);

    BOOST_TEST_EQ(h(0, 0).variance(), 100);
    BOOST_TEST_EQ(h(0, 1).variance(), 1);
    BOOST_TEST_EQ(h(0, 2).variance(), 0);

    BOOST_TEST_EQ(h(1, 0).variance(), 0);
    BOOST_TEST_EQ(h(1, 1).variance(), 0);
    BOOST_TEST_EQ(h(1, 2).variance(), 0);

    BOOST_TEST_EQ(h(2, 0).variance(), 0);
    BOOST_TEST_EQ(h(2, 1).variance(), 0);
    BOOST_TEST_EQ(h(2, 2).variance(), 0);
  }

  // d3w
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 3),
                                              axis::integer<>(0, 4),
                                              axis::integer<>(0, 5));
    for (auto i = 0; i < h.axis(0_c).size(); ++i) {
      for (auto j = 0; j < h.axis(1_c).size(); ++j) {
        for (auto k = 0; k < h.axis(2_c).size(); ++k) {
          h.fill(weight(i + j + k), i, j, k);
        }
      }
    }

    for (auto i = 0; i < h.axis(0_c).size(); ++i) {
      for (auto j = 0; j < h.axis(1_c).size(); ++j) {
        for (auto k = 0; k < h.axis(2_c).size(); ++k) {
          BOOST_TEST_EQ(h(i, j, k).value(), i + j + k);
          BOOST_TEST_EQ(h(i, j, k).variance(), (i + j + k) * (i + j + k));
        }
      }
    }
  }

  // add_1
  {
    auto a = make_histogram<adaptive_storage>(Type(), axis::integer<>(-1, 2));
    auto b =
        make_histogram<array_storage<unsigned>>(Type(), axis::integer<>(-1, 2));
    a.fill(-1);
    b.fill(1);
    auto c = a;
    c += b;
    BOOST_TEST_EQ(c(-1), 0);
    BOOST_TEST_EQ(c(0), 1);
    BOOST_TEST_EQ(c(1), 0);
    BOOST_TEST_EQ(c(2), 1);
    BOOST_TEST_EQ(c(3), 0);
    auto d = a;
    d += b;
    BOOST_TEST_EQ(d(-1), 0);
    BOOST_TEST_EQ(d(0), 1);
    BOOST_TEST_EQ(d(1), 0);
    BOOST_TEST_EQ(d(2), 1);
    BOOST_TEST_EQ(d(3), 0);
  }

  // add_2
  {
    auto a = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2));
    auto b = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2));

    a.fill(0);
    BOOST_TEST_EQ(a(0).variance(), 1);
    b.fill(1, weight(3));
    BOOST_TEST_EQ(b(1).variance(), 9);
    auto c = a;
    c += b;
    BOOST_TEST_EQ(c(-1).value(), 0);
    BOOST_TEST_EQ(c(0).value(), 1);
    BOOST_TEST_EQ(c(0).variance(), 1);
    BOOST_TEST_EQ(c(1).value(), 3);
    BOOST_TEST_EQ(c(1).variance(), 9);
    BOOST_TEST_EQ(c(2).value(), 0);
    auto d = a;
    d += b;
    BOOST_TEST_EQ(d(-1).value(), 0);
    BOOST_TEST_EQ(d(0).value(), 1);
    BOOST_TEST_EQ(d(0).variance(), 1);
    BOOST_TEST_EQ(d(1).value(), 3);
    BOOST_TEST_EQ(d(1).variance(), 9);
    BOOST_TEST_EQ(d(2).value(), 0);
  }

  // add_3
  {
    auto a =
        make_histogram<array_storage<char>>(Type(), axis::integer<>(-1, 2));
    auto b =
        make_histogram<array_storage<unsigned>>(Type(), axis::integer<>(-1, 2));
    a.fill(-1);
    b.fill(1);
    auto c = a;
    c += b;
    BOOST_TEST_EQ(c(-1), 0);
    BOOST_TEST_EQ(c(0), 1);
    BOOST_TEST_EQ(c(1), 0);
    BOOST_TEST_EQ(c(2), 1);
    BOOST_TEST_EQ(c(3), 0);
    auto d = a;
    d += b;
    BOOST_TEST_EQ(d(-1), 0);
    BOOST_TEST_EQ(d(0), 1);
    BOOST_TEST_EQ(d(1), 0);
    BOOST_TEST_EQ(d(2), 1);
    BOOST_TEST_EQ(d(3), 0);
  }

  // bad_add
  {
    auto a = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2));
    auto b = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 3));
    BOOST_TEST_THROWS(a += b, std::logic_error);
  }

  // bad_index
  {
    auto a = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2));
    BOOST_TEST_THROWS(a(5), std::out_of_range);
  }

  // functional programming
  {
    auto v = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 10));
    std::for_each(v.begin(), v.end(), [&h](int x) { h.fill(weight(2.0), x); });
    BOOST_TEST_EQ(h.sum().value(), 20);
  }

  // operators
  {
    auto a = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 3));
    auto b = a;
    a.fill(0);
    b.fill(1);
    auto c = a + b;
    BOOST_TEST_EQ(c(0), 1);
    BOOST_TEST_EQ(c(1), 1);
    c += b;
    BOOST_TEST_EQ(c(0), 1);
    BOOST_TEST_EQ(c(1), 2);
    auto d = a + b + c;
    BOOST_TEST_EQ(d(0), 2);
    BOOST_TEST_EQ(d(1), 3);
    auto e = 3 * a;
    auto f = b * 2;
    BOOST_TEST_EQ(e(0).value(), 3);
    BOOST_TEST_EQ(e(1).value(), 0);
    BOOST_TEST_EQ(f(0).value(), 0);
    BOOST_TEST_EQ(f(1).value(), 2);
    auto r = a;
    r += b;
    r += e;
    BOOST_TEST_EQ(r(0).value(), 4);
    BOOST_TEST_EQ(r(1).value(), 1);
    BOOST_TEST_EQ(r, a + b + 3 * a);
    auto s = r / 4;
    r /= 4;
    BOOST_TEST_EQ(r(0).value(), 1);
    BOOST_TEST_EQ(r(1).value(), 0.25);
    BOOST_TEST_EQ(r, s);
  }

  // histogram_serialization
  {
    enum { A, B, C };
    auto a = make_histogram<adaptive_storage>(
        Type(), axis::regular<>(3, -1, 1, "r"),
        axis::circular<>(4, 0.0, 1.0, "p"),
        axis::regular<double, axis::transform::log>(3, 1, 100, "lr"),
        axis::variable<>({0.1, 0.2, 0.3, 0.4, 0.5}, "v"),
        axis::category<>{A, B, C}, axis::integer<>(0, 2, "i"));
    a.fill(0.5, 20, 0.1, 0.25, 1, 0);
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

  // histogram_ostream
  {
    auto a = make_histogram<adaptive_storage>(
        Type(), axis::regular<>(3, -1, 1, "r"), axis::integer<>(0, 2, "i"));
    std::ostringstream os;
    os << a;
    BOOST_TEST_EQ(os.str(), "histogram("
                            "\n  regular(3, -1, 1, label='r'),"
                            "\n  integer(0, 2, label='i'),"
                            "\n)");
  }

  // histogram_reset
  {
    auto a = make_histogram<adaptive_storage>(
        Type(), axis::integer<>(0, 2, "", axis::uoflow::off));
    a.fill(0);
    a.fill(1);
    BOOST_TEST_EQ(a(0), 1);
    BOOST_TEST_EQ(a(1), 1);
    a.reset();
    BOOST_TEST_EQ(a(0), 0);
    BOOST_TEST_EQ(a(1), 0);
  }

  // reduce
  {
    auto h1 = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2),
                                               axis::integer<>(0, 3));
    h1.fill(0, 0);
    h1.fill(0, 1);
    h1.fill(1, 0);
    h1.fill(1, 1);
    h1.fill(1, 2);

    auto h1_0 = h1.reduce_to(0_c);
    BOOST_TEST_EQ(h1_0.dim(), 1);
    BOOST_TEST_EQ(h1_0.sum(), 5);
    BOOST_TEST_EQ(h1_0(0), 2);
    BOOST_TEST_EQ(h1_0(1), 3);
    BOOST_TEST_EQ(h1_0.axis()[0].lower(), 0);
    BOOST_TEST_EQ(h1_0.axis()[1].lower(), 1);
    BOOST_TEST(axis_equal(Type(), h1_0.axis(), h1.axis(0_c)));

    auto h1_1 = h1.reduce_to(1_c);
    BOOST_TEST_EQ(h1_1.dim(), 1);
    BOOST_TEST_EQ(h1_1.sum(), 5);
    BOOST_TEST_EQ(h1_1(0), 2);
    BOOST_TEST_EQ(h1_1(1), 2);
    BOOST_TEST_EQ(h1_1(2), 1);
    BOOST_TEST(axis_equal(Type(), h1_1.axis(), h1.axis(1_c)));

    auto h2 = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 2),
                                               axis::integer<>(0, 3),
                                               axis::integer<>(0, 4));
    h2.fill(0, 0, 0);
    h2.fill(0, 1, 0);
    h2.fill(0, 1, 1);
    h2.fill(0, 0, 2);
    h2.fill(1, 0, 2);

    auto h2_0 = h2.reduce_to(0_c);
    BOOST_TEST_EQ(h2_0.dim(), 1);
    BOOST_TEST_EQ(h2_0.sum(), 5);
    BOOST_TEST_EQ(h2_0(0), 4);
    BOOST_TEST_EQ(h2_0(1), 1);
    BOOST_TEST(axis_equal(Type(), h2_0.axis(), axis::integer<>(0, 2)));

    auto h2_1 = h2.reduce_to(1_c);
    BOOST_TEST_EQ(h2_1.dim(), 1);
    BOOST_TEST_EQ(h2_1.sum(), 5);
    BOOST_TEST_EQ(h2_1(0), 3);
    BOOST_TEST_EQ(h2_1(1), 2);
    BOOST_TEST(axis_equal(Type(), h2_1.axis(), axis::integer<>(0, 3)));

    auto h2_2 = h2.reduce_to(2_c);
    BOOST_TEST_EQ(h2_2.dim(), 1);
    BOOST_TEST_EQ(h2_2.sum(), 5);
    BOOST_TEST_EQ(h2_2(0), 2);
    BOOST_TEST_EQ(h2_2(1), 1);
    BOOST_TEST_EQ(h2_2(2), 2);
    BOOST_TEST(axis_equal(Type(), h2_2.axis(), axis::integer<>(0, 4)));

    auto h2_01 = h2.reduce_to(0_c, 1_c);
    BOOST_TEST_EQ(h2_01.dim(), 2);
    BOOST_TEST_EQ(h2_01.sum(), 5);
    BOOST_TEST_EQ(h2_01(0, 0), 2);
    BOOST_TEST_EQ(h2_01(0, 1), 2);
    BOOST_TEST_EQ(h2_01(1, 0), 1);
    BOOST_TEST(axis_equal(Type(), h2_01.axis(0_c), axis::integer<>(0, 2)));
    BOOST_TEST(axis_equal(Type(), h2_01.axis(1_c), axis::integer<>(0, 3)));

    auto h2_02 = h2.reduce_to(0_c, 2_c);
    BOOST_TEST_EQ(h2_02.dim(), 2);
    BOOST_TEST_EQ(h2_02.sum(), 5);
    BOOST_TEST_EQ(h2_02(0, 0), 2);
    BOOST_TEST_EQ(h2_02(0, 1), 1);
    BOOST_TEST_EQ(h2_02(0, 2), 1);
    BOOST_TEST_EQ(h2_02(1, 2), 1);
    BOOST_TEST(axis_equal(Type(), h2_02.axis(0_c), axis::integer<>(0, 2)));
    BOOST_TEST(axis_equal(Type(), h2_02.axis(1_c), axis::integer<>(0, 4)));

    auto h2_12 = h2.reduce_to(1_c, 2_c);
    BOOST_TEST_EQ(h2_12.dim(), 2);
    BOOST_TEST_EQ(h2_12.sum(), 5);
    BOOST_TEST_EQ(h2_12(0, 0), 1);
    BOOST_TEST_EQ(h2_12(1, 0), 1);
    BOOST_TEST_EQ(h2_12(1, 1), 1);
    BOOST_TEST_EQ(h2_12(0, 2), 2);
    BOOST_TEST(axis_equal(Type(), h2_12.axis(0_c), axis::integer<>(0, 3)));
    BOOST_TEST(axis_equal(Type(), h2_12.axis(1_c), axis::integer<>(0, 4)));
  }

  // custom axis
  {
    struct custom_axis : public axis::integer<> {
      using value_type = const char *; // type that is fed to the axis

      using integer::integer; // inherit ctors of base

      // the customization point
      // - accept const char* and convert to int
      // - then call index method of base class
      int index(value_type s) const { return integer::index(std::atoi(s)); }
    };

    auto h = make_histogram<adaptive_storage>(Type(), custom_axis(0, 3));
    h.fill("-10");
    h.fill("0");
    h.fill("1");
    h.fill("9");

    BOOST_TEST_EQ(h.dim(), 1);
    BOOST_TEST(h.axis() == custom_axis(0, 3));
    BOOST_TEST_EQ(h(0), 1);
    BOOST_TEST_EQ(h(1), 1);
    BOOST_TEST_EQ(h(2), 0);
  }

  // histogram iterator 1D
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 3));
    const auto &a = h.axis();
    h.fill(0, weight(2));
    h.fill(1);
    h.fill(1);
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
    BOOST_TEST(it == h.end());
  }

  // histogram iterator 2D
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 3),
                                              axis::integer<>(2, 4));
    const auto &a0 = h.axis(0_c);
    const auto &a1 = h.axis(1_c);
    h.fill(0, 2, weight(2));
    h.fill(1, 2);
    h.fill(1, 3);
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
    BOOST_TEST_EQ(*it, 1);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 2);
    BOOST_TEST_EQ(it.idx(1), 0);
    BOOST_TEST_EQ(it.bin(0_c), a0[2]);
    BOOST_TEST_EQ(it.bin(1_c), a1[0]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 0);
    BOOST_TEST_EQ(it.idx(1), 1);
    BOOST_TEST_EQ(it.bin(0_c), a0[0]);
    BOOST_TEST_EQ(it.bin(1_c), a1[1]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 1);
    BOOST_TEST_EQ(it.idx(1), 1);
    BOOST_TEST_EQ(it.bin(0_c), a0[1]);
    BOOST_TEST_EQ(it.bin(1_c), a1[1]);
    BOOST_TEST_EQ(*it, 1);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 2);
    BOOST_TEST_EQ(it.idx(1), 1);
    BOOST_TEST_EQ(it.bin(0_c), a0[2]);
    BOOST_TEST_EQ(it.bin(1_c), a1[1]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST(it == h.end());
  }

  // STL compatibility
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 3));
    for (int i = 0; i < 3; ++i)
      h.fill(i);
    auto a = std::vector<double>();
    std::partial_sum(h.begin(), h.end(), std::back_inserter(a));
    BOOST_TEST_EQ(a[0], 1);
    BOOST_TEST_EQ(a[1], 2);
    BOOST_TEST_EQ(a[2], 3);
  }

  // pass histogram to function
  {
    auto h = make_histogram<adaptive_storage>(Type(), axis::integer<>(0, 3));
    pass_histogram(h);
  }
}

template <typename T1, typename T2> void run_mixed_tests() {

  // compare
  {
    auto a = make_histogram<adaptive_storage>(T1{}, axis::regular<>{3, 0, 3},
                                              axis::integer<>(0, 2));
    auto b = make_histogram<array_storage<int>>(T2{}, axis::regular<>{3, 0, 3},
                                                axis::integer<>(0, 2));
    BOOST_TEST_EQ(a, b);
    auto b2 = make_histogram<adaptive_storage>(T2{}, axis::integer<>{0, 3},
                                               axis::integer<>(0, 2));
    BOOST_TEST_NE(a, b2);
    auto b3 = make_histogram<adaptive_storage>(T2{}, axis::regular<>(3, 0, 4),
                                               axis::integer<>(0, 2));
    BOOST_TEST_NE(a, b3);
  }

  // copy_assign
  {
    auto a = make_histogram<adaptive_storage>(T1{}, axis::regular<>{3, 0, 3},
                                              axis::integer<>(0, 2));
    auto b = make_histogram<array_storage<int>>(T2{}, axis::regular<>{3, 0, 3},
                                                axis::integer<>(0, 2));
    a.fill(1, 1);
    BOOST_TEST_NE(a, b);
    b = a;
    BOOST_TEST_EQ(a, b);
  }
}

int main() {

  // common interface
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  // special stuff that only works with dynamic_tag

  // init
  {
    auto v = std::vector<axis::any<>>();
    v.push_back(axis::regular<>(4, -1, 1));
    v.push_back(axis::integer<>(1, 7));
    auto h = make_dynamic_histogram(v.begin(), v.end());
    BOOST_TEST_EQ(h.axis(0), v[0]);
    BOOST_TEST_EQ(h.axis(1), v[1]);
  }

  // using iterator ranges
  {
    auto h =
        make_dynamic_histogram(axis::integer<>(0, 2), axis::integer<>(2, 4));
    auto v = std::vector<int>(2);
    auto i = std::array<int, 2>();

    v = {0, 2};
    h.fill(v.begin(), v.end());
    v = {1, 3};
    h.fill(v.begin(), v.end());

    i = {{0, 0}};
    BOOST_TEST_EQ(h(i.begin(), i.end()), 1);
    i = {{1, 1}};
    BOOST_TEST_EQ(h(i.begin(), i.end()), 1);

    v = {0, 2};
    h.fill(v.begin(), v.end(), weight(2));
    v = {1, 3};
    h.fill(v.begin(), v.end(), weight(2));

    i = {{0, 0}};
    BOOST_TEST_EQ(h(i.begin(), i.end()).value(), 3);
    i = {{1, 1}};
    BOOST_TEST_EQ(h(i.begin(), i.end()).variance(), 5);
  }

  // axis methods
  {
    enum { A, B };
    auto c = make_dynamic_histogram(axis::category<>({A, B}));
    BOOST_TEST_THROWS(c.axis().lower(0), std::runtime_error);
  }

  // reduce
  {
    auto h1 =
        make_dynamic_histogram(axis::integer<>(0, 2), axis::integer<>(0, 3));
    h1.fill(0, 0);
    h1.fill(0, 1);
    h1.fill(1, 0);
    h1.fill(1, 1);
    h1.fill(1, 2);

    auto h1_0 = h1.reduce_to(0);
    BOOST_TEST_EQ(h1_0.dim(), 1);
    BOOST_TEST_EQ(h1_0.sum(), 5);
    BOOST_TEST_EQ(h1_0(0), 2);
    BOOST_TEST_EQ(h1_0(1), 3);
    BOOST_TEST(axis_equal(dynamic_tag(), h1_0.axis(), h1.axis(0_c)));

    auto h1_1 = h1.reduce_to(1);
    BOOST_TEST_EQ(h1_1.dim(), 1);
    BOOST_TEST_EQ(h1_1.sum(), 5);
    BOOST_TEST_EQ(h1_1(0), 2);
    BOOST_TEST_EQ(h1_1(1), 2);
    BOOST_TEST_EQ(h1_1(2), 1);
    BOOST_TEST(axis_equal(dynamic_tag(), h1_1.axis(), h1.axis(1_c)));
  }

  // histogram iterator
  {
    auto h = make_dynamic_histogram(axis::integer<>(0, 3));
    const auto &a = h.axis();
    h.fill(0, weight(2));
    h.fill(1);
    h.fill(1);
    auto it = h.begin();
    BOOST_TEST_EQ(it.dim(), 1);

    BOOST_TEST_EQ(it.idx(0), 0);
    BOOST_TEST_EQ(it.bin(0), a[0]);
    BOOST_TEST_EQ(it->value(), 2);
    BOOST_TEST_EQ(it->variance(), 4);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 1);
    BOOST_TEST_EQ(it.bin(0), a[1]);
    BOOST_TEST_EQ(*it, 2);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 2);
    BOOST_TEST_EQ(it.bin(0), a[2]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST(it == h.end());
  }

  run_mixed_tests<static_tag, dynamic_tag>();
  run_mixed_tests<dynamic_tag, static_tag>();

  return boost::report_errors();
}
