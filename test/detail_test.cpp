// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/variant.hpp>
#include <boost/mp11.hpp>
#include <tuple>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

using namespace boost::histogram::detail;
using namespace boost::histogram::literals;
namespace mp11 = boost::mp11;

using i0 = mp11::mp_int<0>;
using i1 = mp11::mp_int<1>;
using i2 = mp11::mp_int<2>;
using i3 = mp11::mp_int<3>;

namespace std { // never do this, we only do it to get ADL working
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  os << "[";
  for (const auto& x : v)
    os << x << " ";
  os << "]";
  return os; 
}

struct ostreamer {
  ostream& os;
  template <typename T> void operator()(const T& t) const {
    os << t << " ";
  }
};

template <typename... Ts>
ostream& operator<<(ostream& os, const tuple<Ts...>& t) {
  os << "[";
  ::boost::mp11::tuple_for_each(t, ostreamer{os});
  os << "]";
  return os; 
}
}

int main() {
  // escape0
  {
    std::ostringstream os;
    escape(os, std::string("abc"));
    BOOST_TEST_EQ(os.str(), std::string("'abc'"));
  }

  // escape1
  {
    std::ostringstream os;
    escape(os, std::string("abc\n"));
    BOOST_TEST_EQ(os.str(), std::string("'abc\n'"));
  }

  // escape2
  {
    std::ostringstream os;
    escape(os, std::string("'abc'"));
    BOOST_TEST_EQ(os.str(), std::string("'\\\'abc\\\''"));
  }

  // // assign_axis unreachable branch
  // {
  //   using V1 = boost::variant<float>;
  //   using V2 = boost::variant<int>;
  //   V1 v1(1.0);
  //   V2 v2(2);
  //   boost::apply_visitor(assign_axis<V1>(v1), v2);
  //   BOOST_TEST_EQ(v1, V1(1.0));
  //   BOOST_TEST_EQ(v2, V2(2));
  // }

  // index_mapper 1
  {
    std::vector<unsigned> n{{2, 2}};
    std::vector<bool> b{{true, false}};
    index_mapper m(std::move(n), std::move(b));
    BOOST_TEST_EQ(m.first, 0);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 1);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 2);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 3);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), false);
  }

  // index_mapper 2
  {
    std::vector<unsigned> n{{2, 2}};
    std::vector<bool> b{{false, true}};
    index_mapper m(std::move(n), std::move(b));
    BOOST_TEST_EQ(m.first, 0);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 1);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 2);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 3);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), false);
  }

  // cat
  {
    BOOST_TEST_EQ(cat("foo", 1, "bar"),
                  std::string("foo1bar")); 
  }

  // has_variance_support
  {
    struct no_methods {};

    struct value_method {
      const double &value() const;
    };

    struct variance_method {
      const double &variance() const;
    };

    struct value_and_variance_methods {
      const double &value() const;
      const double &variance() const;
    };

    BOOST_TEST_EQ(has_variance_support<no_methods>(), false);
    BOOST_TEST_EQ(has_variance_support<value_method>(), false);
    BOOST_TEST_EQ(has_variance_support<variance_method>(),
                  false);
    BOOST_TEST_EQ(has_variance_support<value_and_variance_methods>(),
                  true);
  }

  // classify_container
  {
    using result1 = classify_container<int>;
    BOOST_TEST_TRAIT_TRUE(( std::is_same<result1, no_container_tag> ));

    using result1a = classify_container<int&>;
    BOOST_TEST_TRAIT_TRUE(( std::is_same<result1a, no_container_tag> ));

    using result2 = classify_container<std::vector<int>>;
    BOOST_TEST_TRAIT_TRUE(( std::is_same<result2, dynamic_container_tag> ));

    using result2a = classify_container<std::vector<int>&>;
    BOOST_TEST_TRAIT_TRUE(( std::is_same<result2a, dynamic_container_tag> ));

    using result3 = classify_container<std::pair<int, int>>;
    BOOST_TEST_TRAIT_TRUE(( std::is_same<result3, static_container_tag> ));

    using result3a = classify_container<std::pair<int, int>&>;
    BOOST_TEST_TRAIT_TRUE(( std::is_same<result3a, static_container_tag> ));

    using result4 = classify_container<decltype("abc")>;
    BOOST_TEST_TRAIT_TRUE(( std::is_same<result4, dynamic_container_tag> ));
  }

  // bool mask
  {
    auto v1 = bool_mask<i1, i2>(4, false);
    BOOST_TEST_EQ(v1, std::vector<bool>({true, false, false, true}));

    auto v2 = bool_mask<i1, i3>(4, true);
    BOOST_TEST_EQ(v2, std::vector<bool>({false, true, false, true}));
  }

  // rm_cv_ref
  {
    using T1 = int;
    using T2 = const int;
    using T3 = const int&;
    using T4 = volatile int;
    using T5 = volatile const int;
    using T6 = volatile const int&;
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T1>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T2>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T3>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T4>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T5>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T6>, int>));
  }

  // mp_union
  {
    using L1 = mp11::mp_list<int, char, long>;
    using L2 = mp11::mp_list<char, char*>;
    using result = mp_union<L1, L2>;
    using expected = mp11::mp_list<int, char, long, char*>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result, expected>));
  }

  // literals
  {
    auto j0 = 0_c;
    auto j3 = 3_c;
    auto j10 = 10_c;
    auto j213 = 213_c;
    BOOST_TEST_TRAIT_TRUE((std::is_same<i0, decltype(j0)>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<i3, decltype(j3)>));
    BOOST_TEST_EQ(decltype(j10)::value, 10);
    BOOST_TEST_EQ(decltype(j213)::value, 213);
  }

  // selection 
  {
    struct A {};
    struct B {};
    struct C {};

    using input = mp11::mp_list<A, B, C>;
    using result = selection<input, i2, i0>;
    using expected = mp11::mp_list<C, A>;

    BOOST_TEST_TRAIT_TRUE((std::is_same<result, expected>));
  }

  // unique_sorted
  {
    using input = mp11::mp_list_c<int, 3, 3, 1, 2, 1, 2>;
    using result = unique_sorted<input>;
    using expected = mp11::mp_list_c<int, 1, 2, 3>;

    BOOST_TEST_TRAIT_TRUE((std::is_same<result, expected>));
  }

  // make_sub_tuple
  {
    std::tuple<int, long, char> t(1, 2, 3);
    auto result = make_sub_tuple<decltype(t), i1, i2>(t);
    auto expected = std::tuple<long, char>(2, 3);

    BOOST_TEST_EQ(result, expected);
  }

  return boost::report_errors();
}
