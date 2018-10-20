// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/axis/types.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/mp11.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <iterator>
#include "utility.hpp"

namespace bh = boost::histogram;
using namespace bh::detail;
using namespace bh::literals;
namespace mp11 = boost::mp11;

struct VisitorTestFunctor {
  template <typename T>
  T operator()(T&&);
};

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
    struct A {};

    struct B {
      void value() {}
    };

    struct C {
      void variance() {}
    };

    struct D {
      void value() {}
      void variance() {}
    };

    BOOST_TEST_TRAIT_FALSE((has_variance_support<A>));
    BOOST_TEST_TRAIT_FALSE((has_variance_support<B>));
    BOOST_TEST_TRAIT_FALSE((has_variance_support<C>));
    BOOST_TEST_TRAIT_TRUE((has_variance_support<D>));
  }

  // has_method_lower
  {
    struct A {};
    struct B { void lower(int) {} };

    BOOST_TEST_TRAIT_FALSE((has_method_lower<A>));
    BOOST_TEST_TRAIT_TRUE((has_method_lower<B>));
  }

  // has_method_options
  {
    struct NotOptions {};
    struct A {};
    struct B { NotOptions options(); };
    struct C { bh::axis::option_type options(); };

    BOOST_TEST_TRAIT_FALSE((has_method_options<A>));
    BOOST_TEST_TRAIT_FALSE((has_method_options<B>));
    BOOST_TEST_TRAIT_TRUE((has_method_options<C>));
  }

  // has_method_metadata
  {
    struct A {};
    struct B { void metadata(); };

    BOOST_TEST_TRAIT_FALSE((has_method_metadata<A>));
    BOOST_TEST_TRAIT_TRUE((has_method_metadata<B>));
  }

  // is_equal_comparable
  {
    struct A {};
    struct B { bool operator==(const B&); };
    BOOST_TEST_TRAIT_TRUE((is_equal_comparable<int>));
    BOOST_TEST_TRAIT_FALSE((is_equal_comparable<A>));
    BOOST_TEST_TRAIT_TRUE((is_equal_comparable<B>));
  }

  // is_axis
  {
    struct A {};
    struct B { int operator()(double); unsigned size() const; };
    struct C { int operator()(double); };
    struct D { unsigned size(); };
    BOOST_TEST_TRAIT_FALSE((is_axis<A>));
    BOOST_TEST_TRAIT_TRUE((is_axis<B>));
    BOOST_TEST_TRAIT_FALSE((is_axis<C>));
    BOOST_TEST_TRAIT_FALSE((is_axis<D>));
  }

  // is_iterable
  {
    using A = std::vector<int>;
    using B = int[3];
    using C = std::initializer_list<int>;
    BOOST_TEST_TRAIT_FALSE((is_iterable<int>));
    BOOST_TEST_TRAIT_TRUE((is_iterable<A>));
    BOOST_TEST_TRAIT_TRUE((is_iterable<B>));
    BOOST_TEST_TRAIT_TRUE((is_iterable<C>));
  }

  // is_axis_variant
  {
    struct A {};
    BOOST_TEST_TRAIT_FALSE((is_axis_variant<A>));
    BOOST_TEST_TRAIT_TRUE((is_axis_variant<bh::axis::variant<>>));
    BOOST_TEST_TRAIT_TRUE((is_axis_variant<bh::axis::variant<bh::axis::regular<>>>));
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

    using result4 = classify_container<std::string>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<result4, dynamic_container_tag>));

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

  // rm_cvref
  {
    using T1 = int;
    using T2 = int&&;
    using T3 = const int;
    using T4 = const int&;
    using T5 = volatile int;
    using T6 = volatile int&&;
    using T7 = volatile const int;
    using T8 = volatile const int&;
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cvref<T1>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cvref<T2>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cvref<T3>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cvref<T4>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cvref<T5>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cvref<T6>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cvref<T7>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<rm_cvref<T8>, int>));
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

  // mp_last
  {
    using L = mp11::mp_list<int, char, long>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<mp_last<L>, long>));
  }

  // container_element_type
  {
    using T1 = std::vector<int>;
    using U1 = container_element_type<T1>;
    using T2 = const std::vector<const int>&;
    using U2 = container_element_type<T2>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<U1, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<U2, const int>));
  }

  // unqualified_iterator_value_type
  {
    using T1 = const char*;
    using T2 = std::iterator<std::random_access_iterator_tag, int>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqualified_iterator_value_type<T1>, char>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqualified_iterator_value_type<T2>, int>));
  }

  // visitor_return_type
  {
    using V1 = bh::axis::variant<char>;
    using V2 = bh::axis::variant<int>&;
    using V3 = const bh::axis::variant<long>&;
    BOOST_TEST_TRAIT_TRUE((std::is_same<visitor_return_type<VisitorTestFunctor, V1>, char>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<visitor_return_type<VisitorTestFunctor, V2>, int&>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<visitor_return_type<VisitorTestFunctor, V3>, const long&>));
  }

  // is_axis_vector
  {
    using A = std::vector<bh::axis::regular<>>;
    using B = std::vector<bh::axis::variant<bh::axis::regular<>>>;
    using C = std::vector<int>;
    using D = bh::axis::regular<>;
    BOOST_TEST_TRAIT_TRUE((is_axis_vector<A>));
    BOOST_TEST_TRAIT_TRUE((is_axis_vector<B>));
    BOOST_TEST_TRAIT_FALSE((is_axis_vector<C>));
    BOOST_TEST_TRAIT_FALSE((is_axis_vector<D>));
  }

  return boost::report_errors();
}
