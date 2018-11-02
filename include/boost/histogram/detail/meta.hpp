// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_META_HPP
#define BOOST_HISTOGRAM_DETAIL_META_HPP

#include <boost/callable_traits/args.hpp>
#include <boost/callable_traits/return_type.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/mp11.hpp>
#include <functional>
#include <iterator>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

template <class T>
using unqual = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T>
using mp_size = mp11::mp_size<unqual<T>>;

template <typename T, unsigned N>
using mp_at_c = mp11::mp_at_c<unqual<T>, N>;

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
using mp_last = mp11::mp_at_c<L, (mp_size<L>::value - 1)>;

template <typename T>
using container_value_type = mp11::mp_first<unqual<T>>;

template <typename T>
using iterator_value_type = typename std::iterator_traits<T>::value_type;

template <typename T>
using return_type = typename boost::callable_traits::return_type<T>::type;

template <typename T>
using args_type = mp11::mp_if<std::is_member_function_pointer<T>,
                              mp11::mp_pop_front<boost::callable_traits::args_t<T>>,
                              boost::callable_traits::args_t<T>>;

template <typename T, std::size_t N = 0>
using arg_type = typename mp11::mp_at_c<args_type<T>, N>;

template <typename F, typename V>
using visitor_return_type =
    decltype(std::declval<F>()(std::declval<copy_qualifiers<V, mp_at_c<V, 0>>>()));

template <bool B, typename T, typename F, typename... Ts>
constexpr decltype(auto) static_if_c(T&& t, F&& f, Ts&&... ts) {
  return std::get<(B ? 0 : 1)>(std::forward_as_tuple(
      std::forward<T>(t), std::forward<F>(f)))(std::forward<Ts>(ts)...);
}

template <typename B, typename... Ts>
constexpr decltype(auto) static_if(Ts&&... ts) {
  return static_if_c<B::value>(std::forward<Ts>(ts)...);
}

#define BOOST_HISTOGRAM_MAKE_SFINAE(name, cond)      \
  template <typename U>                              \
  struct name##_impl {                               \
    template <typename T, typename = decltype(cond)> \
    static std::true_type Test(void*);               \
    template <typename T>                            \
    static std::false_type Test(...);                \
    using type = decltype(Test<U>(nullptr));         \
  };                                                 \
  template <typename T>                              \
  using name = typename name##_impl<T>::type

BOOST_HISTOGRAM_MAKE_SFINAE(has_variance_support,
                            (std::declval<T&>().value(), std::declval<T&>().variance()));

BOOST_HISTOGRAM_MAKE_SFINAE(has_method_value, (std::declval<T&>().value(0)));

// TODO try casting to more specific pmf with correct return type
BOOST_HISTOGRAM_MAKE_SFINAE(has_method_options, (std::declval<T&>().options()));

BOOST_HISTOGRAM_MAKE_SFINAE(has_method_metadata, (std::declval<T&>().metadata()));

// resize has two overloads, trying to get pmf in this case always fails
BOOST_HISTOGRAM_MAKE_SFINAE(has_method_resize, (std::declval<T&>().resize(0)));

BOOST_HISTOGRAM_MAKE_SFINAE(has_method_size, &T::size);

BOOST_HISTOGRAM_MAKE_SFINAE(has_method_clear, &T::clear);

BOOST_HISTOGRAM_MAKE_SFINAE(has_allocator, &T::get_allocator);

BOOST_HISTOGRAM_MAKE_SFINAE(is_indexable, (std::declval<T&>()[0]));

BOOST_HISTOGRAM_MAKE_SFINAE(is_transform, (&T::forward, &T::inverse));

BOOST_HISTOGRAM_MAKE_SFINAE(is_storage, (typename T::storage_tag()));

BOOST_HISTOGRAM_MAKE_SFINAE(is_vector_like,
                            (std::declval<T&>()[0], &T::size,
                             std::declval<T&>().resize(0), &T::cbegin, &T::cend));

BOOST_HISTOGRAM_MAKE_SFINAE(is_array_like,
                            (std::declval<T&>()[0], &T::size, std::tuple_size<T>::value,
                             &T::cbegin, &T::cend));

BOOST_HISTOGRAM_MAKE_SFINAE(is_map_like,
                            (typename T::key_type(), typename T::mapped_type(),
                             std::declval<T&>().begin(), std::declval<T&>().end()));

BOOST_HISTOGRAM_MAKE_SFINAE(is_indexable_container, (std::declval<T&>()[0], &T::size,
                                                     std::begin(std::declval<T&>()),
                                                     std::end(std::declval<T&>())));

BOOST_HISTOGRAM_MAKE_SFINAE(is_tuple, (std::get<0>(std::declval<T&>())));

BOOST_HISTOGRAM_MAKE_SFINAE(is_equal_comparable,
                            (std::declval<T&>() == std::declval<T&>()));

BOOST_HISTOGRAM_MAKE_SFINAE(is_axis, (&T::size, &T::operator()));

BOOST_HISTOGRAM_MAKE_SFINAE(is_iterable, (std::begin(std::declval<T&>()),
                                          std::end(std::declval<T&>())));

BOOST_HISTOGRAM_MAKE_SFINAE(is_streamable,
                            (std::declval<std::ostream&>() << std::declval<T&>()));

namespace {
template <typename T>
struct is_axis_variant_impl : std::false_type {};

template <typename... Ts>
struct is_axis_variant_impl<axis::variant<Ts...>> : std::true_type {};
} // namespace

template <typename T>
using is_axis_variant = typename is_axis_variant_impl<T>::type;

template <typename T>
using is_axis_or_axis_variant = mp11::mp_or<is_axis<T>, is_axis_variant<T>>;

template <typename T, typename U = container_value_type<T>>
using is_axis_vector =
    mp11::mp_all<is_indexable_container<unqual<T>>, is_axis_or_axis_variant<U>>;

struct static_container_tag {};
struct iterable_container_tag {};
struct no_container_tag {};

template <typename T>
using classify_container = typename std::conditional<
    is_iterable<T>::value, iterable_container_tag,
    typename std::conditional<is_tuple<T>::value, static_container_tag,
                              no_container_tag>::type>::type;

namespace {
struct bool_mask_impl {
  std::vector<bool>& b;
  bool v;
  template <typename Int>
  void operator()(Int) const {
    b[Int::value] = v;
  }
};
} // namespace

template <typename... Ns>
std::vector<bool> bool_mask(unsigned n, bool v) {
  std::vector<bool> b(n, !v);
  mp11::mp_for_each<mp11::mp_list<Ns...>>(bool_mask_impl{b, v});
  return b;
}

// poor-mans concept checks
template <typename T, typename = decltype(*std::declval<T&>(), ++std::declval<T&>())>
struct requires_iterator {};

template <typename T, typename = mp11::mp_if<is_iterable<T>, void>>
struct requires_iterable {};

template <typename T, typename = mp11::mp_if<is_tuple<T>, void>>
struct requires_tuple {};

template <typename T, typename = mp11::mp_if<is_axis<T>, void>>
struct requires_axis {};

template <typename T, typename = mp11::mp_if<is_axis_or_axis_variant<T>, void>>
struct requires_axis_or_axis_variant {};

template <typename T, typename = mp11::mp_if<is_axis_vector<T>, void>>
struct requires_axis_vector {};

template <typename T, typename U, typename = mp11::mp_if<std::is_same<T, U>, void>>
struct requires_same {};

template <typename T, typename = mp11::mp_if<is_transform<T>, void>>
struct requires_transform {};

template <typename T, typename = mp11::mp_if<is_storage<T>, void>>
struct requires_storage {};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
