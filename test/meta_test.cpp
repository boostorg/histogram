// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/mp11.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "utility.hpp"

using namespace boost::histogram::detail;
using namespace boost::histogram::literals;
namespace mp11 = boost::mp11;

int main() {
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

  // has_variance_support
  {
    struct no_methods {};

    struct value_method {
      void value() {}
    };

    struct variance_method {
      void variance() {}
    };

    struct value_and_variance_methods {
      void value() {}
      void variance() {}
    };

    BOOST_TEST_TRAIT_FALSE((has_variance_support<no_methods>));
    BOOST_TEST_TRAIT_FALSE((has_variance_support<value_method>));
    BOOST_TEST_TRAIT_FALSE((has_variance_support<variance_method>));
    BOOST_TEST_TRAIT_TRUE((has_variance_support<value_and_variance_methods>));
  }

  // has_method_lower
  {
    struct no_methods {};
    struct lower_method {
      void lower(int) {}
    };

    BOOST_TEST_TRAIT_FALSE((has_method_lower<no_methods>));
    BOOST_TEST_TRAIT_TRUE((has_method_lower<lower_method>));
  }

  // classify_container
  {
    using result1 = classify_container<int>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result1, no_container_tag>));

    using result1a = classify_container<int&>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result1a, no_container_tag>));

    using result2 = classify_container<std::vector<int>>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result2, dynamic_container_tag>));

    using result2a = classify_container<std::vector<int>&>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result2a, dynamic_container_tag>));

    using result3 = classify_container<std::pair<int, int>>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result3, static_container_tag>));

    using result3a = classify_container<std::pair<int, int>&>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result3a, static_container_tag>));

    // (c-)strings are not regarded as dynamic containers
    using result4a = classify_container<decltype("abc")>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result4a, no_container_tag>));

    using result4b = classify_container<std::string>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result4b, no_container_tag>));

    using result5 = classify_container<int*>; // has no std::end
    BOOST_TEST_TRAIT_TRUE((std::is_same<result5, no_container_tag>));
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
    using T2 = int&&;
    using T3 = const int;
    using T4 = const int&;
    using T5 = volatile int;
    using T6 = volatile int&&;
    using T7 = volatile const int;
    using T8 = volatile const int&;
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T1>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T2>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T3>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T4>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T5>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T6>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T7>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cv_ref<T8>, int>));
  }

  // mp_size
  {
    using T1 = std::tuple<int>;
    using T2 = const std::tuple<int, int&>;
    using T3 = std::tuple<int, int&, int*>&;
    using T4 = const std::tuple<int, int&, int*, volatile int>&;
    BOOST_TEST_EQ(mp_size<T1>::value, 1);
    BOOST_TEST_EQ(mp_size<T2>::value, 2);
    BOOST_TEST_EQ(mp_size<T3>::value, 3);
    BOOST_TEST_EQ(mp_size<T4>::value, 4);
  }

  // copy_qualifiers
  {
    BOOST_TEST_TRAIT_TRUE((std::is_same<copy_qualifiers<int, long>, long>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<copy_qualifiers<const int, long>, const long>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<copy_qualifiers<int&, long>, long&>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<copy_qualifiers<const int&, long>, const long&>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<copy_qualifiers<int&&, long>, long&&>));
  }

  // mp_set_union
  {
    using L1 = mp11::mp_list<int, char, long>;
    using L2 = mp11::mp_list<char, int, char, char*>;
    using result = mp_set_union<L1, L2>;
    using expected = mp11::mp_list<int, char, long, char*>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result, expected>));
  }

  return boost::report_errors();
}
