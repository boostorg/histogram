// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_META_HPP
#define BOOST_HISTOGRAM_DETAIL_META_HPP

#include <boost/mp11.hpp>
#include <iterator>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

#define BOOST_HISTOGRAM_MAKE_SFINAE(name, cond)      \
  template <typename U>                              \
  struct name##_impl {                               \
    template <typename T, typename = decltype(cond)> \
    struct SFINAE {};                                \
    template <typename T>                            \
    static std::true_type Test(SFINAE<T>*);          \
    template <typename T>                            \
    static std::false_type Test(...);                \
    using type = decltype(Test<U>(nullptr));         \
  };                                                 \
  template <typename T>                              \
  using name = typename name##_impl<T>::type

BOOST_HISTOGRAM_MAKE_SFINAE(has_variance_support,
                            (std::declval<T&>().value(), std::declval<T&>().variance()));

BOOST_HISTOGRAM_MAKE_SFINAE(has_method_lower, (std::declval<T&>().lower(0)));

BOOST_HISTOGRAM_MAKE_SFINAE(is_dynamic_container, (std::begin(std::declval<T&>())));

BOOST_HISTOGRAM_MAKE_SFINAE(is_static_container, (std::get<0>(std::declval<T&>())));

BOOST_HISTOGRAM_MAKE_SFINAE(is_castable_to_int, (static_cast<int>(std::declval<T&>())));

BOOST_HISTOGRAM_MAKE_SFINAE(is_string, (std::declval<T&>().c_str()));

struct static_container_tag {};
struct dynamic_container_tag {};
struct no_container_tag {};

template <typename T>
using classify_container = typename std::conditional<
    is_static_container<T>::value, static_container_tag,
    typename std::conditional<(is_dynamic_container<T>::value &&
                               !std::is_convertible<T, const char*>::value &&
                               !is_string<T>::value),
                              dynamic_container_tag, no_container_tag>::type>::type;

template <typename T,
          typename = decltype(std::declval<T&>().size(), std::declval<T&>().increase(0),
                              std::declval<T&>()[0])>
struct requires_storage {};

template <typename T, typename = decltype(*std::declval<T&>(), ++std::declval<T&>())>
struct requires_iterator {};

template <typename T, typename = decltype(std::declval<T&>()[0])>
struct requires_vector {};

template <typename T, typename = decltype(std::get<0>(std::declval<T&>()))>
struct requires_tuple {};

template <typename T>
using requires_axis = decltype(std::declval<T&>().size(), std::declval<T&>().shape(),
                               std::declval<T&>().uoflow(), std::declval<T&>().label(),
                               std::declval<T&>()[0]);

namespace {
struct bool_mask_impl {
  std::vector<bool>& b;
  bool v;
  template <typename Int>
  void operator()(Int) const {
    b[Int::value] = v;
  }
};
}

template <typename... Ns>
std::vector<bool> bool_mask(unsigned n, bool v) {
  std::vector<bool> b(n, !v);
  mp11::mp_for_each<mp11::mp_list<Ns...>>(bool_mask_impl{b, v});
  return b;
}

template <class T>
using rm_cv_ref = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template <class T>
using mp_size = mp11::mp_size<rm_cv_ref<T>>;

template <typename T, unsigned D>
using mp_at_c = mp11::mp_at_c<rm_cv_ref<T>, D>;

template <typename T1, typename T2>
using copy_qualifiers = mp11::mp_if<
    std::is_rvalue_reference<T1>, T2&&,
    mp11::mp_if<std::is_lvalue_reference<T1>,
                mp11::mp_if<std::is_const<typename std::remove_reference<T1>::type>,
                            const T2&, T2&>,
                mp11::mp_if<std::is_const<T1>, const T2, T2>>>;

template <typename S, typename L>
using mp_set_union = mp11::mp_apply_q<mp11::mp_bind_front<mp11::mp_set_push_back, S>, L>;

template <typename L>
using mp_last = mp11::mp_at_c<L, (mp11::mp_size<L>::value - 1)>;

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
