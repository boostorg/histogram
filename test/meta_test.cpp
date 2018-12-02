// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/adaptive_storage.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/sample.hpp>
#include <boost/histogram/weight.hpp>
#include <boost/mp11.hpp>
#include <deque>
#include <iterator>
#include <map>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "utility_meta.hpp"

namespace bh = boost::histogram;
using namespace bh::detail;
using namespace bh::literals;
namespace mp11 = boost::mp11;

struct VisitorTestFunctor {
  template <typename T>
  T operator()(T&&);
};

int main() {
  // has_method_value
  {
    struct A {};
    struct B {
      A value(int) const { return {}; }
    };
    struct C {
      char value(int) const { return 0; }
    };

    BOOST_TEST_TRAIT_FALSE((has_method_value<A, double>));
    BOOST_TEST_TRAIT_TRUE((has_method_value<B, A>));
    BOOST_TEST_TRAIT_FALSE((has_method_value<B, char>));
    BOOST_TEST_TRAIT_TRUE((has_method_value<C, char>));
    BOOST_TEST_TRAIT_FALSE((has_method_value<C, A>));
  }

  // has_method_options
  {
    struct A {};
    struct B {
      void options() {}
    };
    struct C {
      bh::axis::option_type options() const { return {}; }
    };

    BOOST_TEST_TRAIT_FALSE((has_method_options<A>));
    BOOST_TEST_TRAIT_FALSE((has_method_options<B>));
    BOOST_TEST_TRAIT_TRUE((has_method_options<C>));
  }

  // has_method_metadata
  {
    struct A {};
    struct B {
      void metadata();
    };

    BOOST_TEST_TRAIT_FALSE((has_method_metadata<A>));
    BOOST_TEST_TRAIT_TRUE((has_method_metadata<B>));
  }

  // has_method_resize
  {
    struct A {};
    using B = std::vector<int>;
    using C = std::map<int, int>;

    BOOST_TEST_TRAIT_FALSE((has_method_resize<A>));
    BOOST_TEST_TRAIT_TRUE((has_method_resize<B>));
    BOOST_TEST_TRAIT_FALSE((has_method_resize<C>));
  }

  // has_method_size
  {
    struct A {};
    using B = std::vector<int>;
    using C = std::map<int, int>;

    BOOST_TEST_TRAIT_FALSE((has_method_size<A>));
    BOOST_TEST_TRAIT_TRUE((has_method_size<B>));
    BOOST_TEST_TRAIT_TRUE((has_method_size<C>));
  }

  // has_method_clear
  {
    struct A {};
    using B = std::vector<int>;
    using C = std::map<int, int>;
    using D = std::array<int, 10>;

    BOOST_TEST_TRAIT_FALSE((has_method_clear<A>));
    BOOST_TEST_TRAIT_TRUE((has_method_clear<B>));
    BOOST_TEST_TRAIT_TRUE((has_method_clear<C>));
    BOOST_TEST_TRAIT_FALSE((has_method_clear<D>));
  }

  // has_allocator
  {
    struct A {};
    using B = std::vector<int>;
    using C = std::map<int, int>;
    using D = std::array<int, 10>;

    BOOST_TEST_TRAIT_FALSE((has_method_clear<A>));
    BOOST_TEST_TRAIT_TRUE((has_method_clear<B>));
    BOOST_TEST_TRAIT_TRUE((has_method_clear<C>));
    BOOST_TEST_TRAIT_FALSE((has_method_clear<D>));
  }

  // is_storage
  {
    struct A {};
    using B = bh::adaptive_storage<>;

    BOOST_TEST_TRAIT_FALSE((is_storage<A>));
    BOOST_TEST_TRAIT_TRUE((is_storage<B>));
  }

  // is_indexable
  {
    struct A {};
    using B = std::vector<int>;
    using C = std::map<int, int>;
    using D = std::map<A, int>;

    BOOST_TEST_TRAIT_FALSE((is_indexable<A>));
    BOOST_TEST_TRAIT_TRUE((is_indexable<B>));
    BOOST_TEST_TRAIT_TRUE((is_indexable<C>));
    BOOST_TEST_TRAIT_FALSE((is_indexable<D>));
  }

  // is_transform
  {
    struct A {};
    struct B {
      double forward(double);
      double inverse(double);
    };

    BOOST_TEST_TRAIT_FALSE((is_transform<A>));
    BOOST_TEST_TRAIT_TRUE((is_transform<B>));
  }

  // is_vector_like
  {
    struct A {};
    using B = std::vector<int>;
    using C = std::array<int, 10>;
    using D = std::map<unsigned, int>;
    using E = std::deque<int>;
    BOOST_TEST_TRAIT_FALSE((is_vector_like<A>));
    BOOST_TEST_TRAIT_TRUE((is_vector_like<B>));
    BOOST_TEST_TRAIT_FALSE((is_vector_like<C>));
    BOOST_TEST_TRAIT_FALSE((is_vector_like<D>));
    BOOST_TEST_TRAIT_TRUE((is_vector_like<E>));
  }

  // is_array_like
  {
    struct A {};
    using B = std::vector<int>;
    using C = std::array<int, 10>;
    using D = std::map<unsigned, int>;
    BOOST_TEST_TRAIT_FALSE((is_array_like<A>));
    BOOST_TEST_TRAIT_FALSE((is_array_like<B>));
    BOOST_TEST_TRAIT_TRUE((is_array_like<C>));
    BOOST_TEST_TRAIT_FALSE((is_array_like<D>));
  }

  // is_map_like
  {
    struct A {};
    using B = std::vector<int>;
    using C = std::array<int, 10>;
    using D = std::map<unsigned, int>;
    using E = std::unordered_map<unsigned, int>;
    BOOST_TEST_TRAIT_FALSE((is_map_like<A>));
    BOOST_TEST_TRAIT_FALSE((is_map_like<B>));
    BOOST_TEST_TRAIT_FALSE((is_map_like<C>));
    BOOST_TEST_TRAIT_TRUE((is_map_like<D>));
    BOOST_TEST_TRAIT_TRUE((is_map_like<E>));
  }

  // is_equal_comparable
  {
    struct A {};
    struct B {
      bool operator==(const B&);
    };
    BOOST_TEST_TRAIT_TRUE((is_equal_comparable<int>));
    BOOST_TEST_TRAIT_FALSE((is_equal_comparable<A>));
    BOOST_TEST_TRAIT_TRUE((is_equal_comparable<B>));
  }

  // is_axis
  {
    struct A {};
    struct B {
      int operator()(double);
      unsigned size() const;
    };
    struct C {
      int operator()(double);
    };
    struct D {
      unsigned size();
    };
    using E = bh::axis::variant<bh::axis::regular<>>;

    BOOST_TEST_TRAIT_FALSE((is_axis<A>));
    BOOST_TEST_TRAIT_TRUE((is_axis<B>));
    BOOST_TEST_TRAIT_FALSE((is_axis<C>));
    BOOST_TEST_TRAIT_FALSE((is_axis<D>));
    BOOST_TEST_TRAIT_FALSE((is_axis<E>));
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

  // is_streamable
  {
    struct Foo {};
    BOOST_TEST_TRAIT_TRUE((is_streamable<int>));
    BOOST_TEST_TRAIT_TRUE((is_streamable<std::string>));
    BOOST_TEST_TRAIT_FALSE((is_streamable<Foo>));
  }

  // is_axis_variant
  {
    struct A {};
    BOOST_TEST_TRAIT_FALSE((is_axis_variant<A>));
    BOOST_TEST_TRAIT_TRUE((is_axis_variant<bh::axis::variant<>>));
    BOOST_TEST_TRAIT_TRUE((is_axis_variant<bh::axis::variant<bh::axis::regular<>>>));
  }

  // unqual
  {
    using T1 = int;
    using T2 = int&&;
    using T3 = const int;
    using T4 = const int&;
    using T5 = volatile int;
    using T6 = volatile int&&;
    using T7 = volatile const int;
    using T8 = volatile const int&;
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqual<T1>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqual<T2>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqual<T3>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqual<T4>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqual<T5>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqual<T6>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqual<T7>, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<unqual<T8>, int>));
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

  // container_value_type
  {
    using T1 = std::vector<int>;
    using U1 = container_value_type<T1>;
    using T2 = const std::vector<const int>&;
    using U2 = container_value_type<T2>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<U1, int>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<U2, const int>));
  }

  // iterator_value_type
  {
    using T1 = const char*;
    using T2 = std::iterator<std::random_access_iterator_tag, int>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<iterator_value_type<T1>, char>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<iterator_value_type<T2>, int>));
  }

  // args_type
  {
    struct Foo {
      static int f1(char);
      int f2(long) const;
    };

    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<args_type<decltype(&Foo::f1)>, std::tuple<char>>));
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<args_type<decltype(&Foo::f2)>, std::tuple<long>>));
  }

  // visitor_return_type
  {
    using V1 = bh::axis::variant<char>;
    using V2 = bh::axis::variant<int>&;
    using V3 = const bh::axis::variant<long>&;
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<visitor_return_type<VisitorTestFunctor, V1>, char>));
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<visitor_return_type<VisitorTestFunctor, V2>, int&>));
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<visitor_return_type<VisitorTestFunctor, V3>, const long&>));
  }

  // static_if
  {
    struct callable {
      int operator()() { return 1; };
    };
    struct not_callable {};
    auto fcn = [](auto b, auto x) {
      return static_if<decltype(b)>([](auto x) { return x(); }, [](auto) { return 2; },
                                    x);
    };
    BOOST_TEST_EQ(fcn(std::true_type(), callable()), 1);
    BOOST_TEST_EQ(fcn(std::false_type(), not_callable()), 2);
  }

  // sub_tuple
  {
    auto a = std::make_tuple(1, 2, 3, 4);
    auto b = sub_tuple<1, 2>(a);
    BOOST_TEST_EQ(b, std::make_tuple(2, 3));
  }

  // is_axis_vector
  {
    using A = std::vector<bh::axis::regular<>>;
    using B = std::vector<bh::axis::variant<bh::axis::regular<>>>;
    using C = std::vector<int>;
    using D = const std::vector<bh::axis::variant<bh::axis::integer<>>>;
    using E = const std::vector<bh::axis::variant<bh::axis::integer<>>>&;
    auto v = std::vector<bh::axis::variant<bh::axis::regular<>, bh::axis::integer<>>>();
    BOOST_TEST_TRAIT_TRUE((is_axis_vector<A>));
    BOOST_TEST_TRAIT_TRUE((is_axis_vector<B>));
    BOOST_TEST_TRAIT_FALSE((is_axis_vector<C>));
    BOOST_TEST_TRAIT_TRUE((is_axis_vector<D>));
    BOOST_TEST_TRAIT_TRUE((is_axis_vector<E>));
    BOOST_TEST_TRAIT_TRUE((is_axis_vector<decltype(v)>));
    BOOST_TEST_TRAIT_TRUE((is_axis_vector<decltype(std::move(v))>));
  }

  // is_weight
  {
    struct A {};
    using B = int;
    using C = decltype(bh::weight(1));
    BOOST_TEST_TRAIT_FALSE((is_weight<A>));
    BOOST_TEST_TRAIT_FALSE((is_weight<B>));
    BOOST_TEST_TRAIT_TRUE((is_weight<C>));
  }

  // is_sample
  {
    struct A {};
    using B = int;
    using C = decltype(bh::sample(1));
    using D = decltype(bh::sample(1, 2.0));
    BOOST_TEST_TRAIT_FALSE((is_sample<A>));
    BOOST_TEST_TRAIT_FALSE((is_sample<B>));
    BOOST_TEST_TRAIT_TRUE((is_sample<C>));
    BOOST_TEST_TRAIT_TRUE((is_sample<D>));
  }

  // has_fixed_size
  {
    struct A {};
    using B = std::vector<int>;
    using C = std::tuple<int, int>;
    using D = std::array<int, 2>;

    BOOST_TEST_NOT(has_fixed_size<A>::value);
    BOOST_TEST_NOT(has_fixed_size<B>::value);
    BOOST_TEST(has_fixed_size<C>::value);
    BOOST_TEST(has_fixed_size<D>::value);
  }

  return boost::report_errors();
}
