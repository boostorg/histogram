// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_META_HPP_
#define _BOOST_HISTOGRAM_DETAIL_META_HPP_

#include <boost/fusion/algorithm/iteration/fold.hpp>
#include <boost/fusion/include/fold.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/unique.hpp>
#include <boost/mpl/vector.hpp>

#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>
#include <utility>

namespace boost {
namespace histogram {
namespace detail {

#define BOOST_HISTOGRAM_MAKE_SFINAE(name, cond)                  \
template <typename U> struct name {                              \
  template <typename T, typename = decltype(cond)>               \
  struct SFINAE {};                                              \
  template <typename T> static std::true_type Test(SFINAE<T> *); \
  template <typename T> static std::false_type Test(...);        \
  using type = decltype(Test<U>(nullptr));                       \
};                                                               \
template <typename T>                                            \
using name##_t = typename name<T>::type

BOOST_HISTOGRAM_MAKE_SFINAE(has_variance_support,
                            (std::declval<T&>().value(), std::declval<T&>().variance()));

BOOST_HISTOGRAM_MAKE_SFINAE(has_method_lower,
                            (std::declval<T &>().lower(0)));

BOOST_HISTOGRAM_MAKE_SFINAE(is_dynamic_container,
                            (std::begin(std::declval<T&>())));

BOOST_HISTOGRAM_MAKE_SFINAE(is_static_container,
                            (std::get<0>(std::declval<T&>())));

BOOST_HISTOGRAM_MAKE_SFINAE(is_castable_to_int,
                            (static_cast<int>(std::declval<T&>())));

struct static_container_tag {};
struct dynamic_container_tag {};
struct no_container_tag {};

template <typename T>
using classify_container_t =
  typename std::conditional<
    is_static_container_t<T>::value,
    static_container_tag,
    typename std::conditional<
      is_dynamic_container_t<T>::value,
      dynamic_container_tag,
      no_container_tag
    >::type
  >::type;

template <typename T, typename = decltype(std::declval<T &>().size(),
                                          std::declval<T &>().increase(0),
                                          std::declval<T &>()[0])>
struct requires_storage {};

template <typename T,
          typename = decltype(*std::declval<T &>(), ++std::declval<T &>())>
struct requires_iterator {};

template <typename MainVector, typename AuxVector>
struct union_
    : mpl::copy_if<AuxVector, mpl::not_<mpl::contains<MainVector, mpl::_1>>,
                   mpl::back_inserter<MainVector>> {};

template <typename MainVector, typename AuxVector>
using union_t = typename union_<MainVector, AuxVector>::type;

struct bool_mask_op {
  std::vector<bool> &b;
  bool v;
  template <typename N> void operator()(const N &) const { b[N::value] = v; }
};

template <typename Ns> std::vector<bool> bool_mask(unsigned n, bool v) {
  std::vector<bool> b(n, !v);
  mpl::for_each<Ns>(bool_mask_op{b, v});
  return b;
}

template <typename Axes, typename Ns> struct axes_assign_subset_op {
  const Axes &axes_;
  template <int N, typename R>
  auto operator()(mpl::int_<N>, R &r) const -> mpl::int_<N + 1> {
    using I2 = typename mpl::at_c<Ns, N>::type;
    r = fusion::at_c<I2::value>(axes_);
    return {};
  }
};

template <typename Ns, typename Axes1, typename Axes>
void axes_assign_subset(Axes1 &axes1, const Axes &axes) {
  fusion::fold(axes1, mpl::int_<0>(), axes_assign_subset_op<Axes, Ns>{axes});
}

template <typename Ns>
using unique_sorted_t =
    typename mpl::unique<typename mpl::sort<Ns>::type,
                         std::is_same<mpl::_1, mpl::_2>>::type;

template <typename Axes, typename Numbers>
using axes_select_t =
    typename mpl::transform<Numbers, mpl::at<Axes, mpl::_>>::type;

template <typename T>
using size_of = std::tuple_size<typename std::decay<T>::type>;

template <unsigned D, typename T>
using type_of = typename std::tuple_element<D, typename std::decay<T>::type>::type;

template <bool C, typename T1, typename T2>
using if_else = typename std::conditional<C, T1, T2>::type;

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
