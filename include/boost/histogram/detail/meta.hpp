// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_META_HPP
#define BOOST_HISTOGRAM_DETAIL_META_HPP

#include <boost/config/workaround.hpp>
#if BOOST_WORKAROUND(BOOST_GCC, >= 60000)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnoexcept-type"
#endif
#include <boost/callable_traits/args.hpp>
#if BOOST_WORKAROUND(BOOST_GCC, >= 60000)
#pragma GCC diagnostic pop
#endif
#include <array>
#include <boost/histogram/detail/static_if.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/function.hpp>
#include <boost/mp11/integer_sequence.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/utility.hpp>
#include <functional>
#include <iterator>
#include <limits>
#include <tuple>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <class T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T, template <class> class... F>
using mp_eval_and = mp11::mp_and<F<T>...>;

template <class T>
struct unref_impl {
  using type = T;
};

template <class T>
struct unref_impl<std::reference_wrapper<T>> {
  using type = T;
};

template <class T>
using unref_t = typename unref_impl<T>::type;

template <class T, class U>
using convert_integer = mp11::mp_if<std::is_integral<remove_cvref_t<T>>, U, T>;

template <class T1, class T2>
using copy_qualifiers = mp11::mp_if<
    std::is_rvalue_reference<T1>, T2&&,
    mp11::mp_if<std::is_lvalue_reference<T1>,
                mp11::mp_if<std::is_const<typename std::remove_reference<T1>::type>,
                            const T2&, T2&>,
                mp11::mp_if<std::is_const<T1>, const T2, T2>>>;

template <class T, class Args = boost::callable_traits::args_t<T>>
using args_type =
    mp11::mp_if<std::is_member_function_pointer<T>, mp11::mp_pop_front<Args>, Args>;

template <class T, std::size_t N = 0>
using arg_type = typename mp11::mp_at_c<args_type<T>, N>;

template <typename T>
constexpr T lowest() {
  return std::numeric_limits<T>::lowest();
}

template <>
constexpr double lowest() {
  return -std::numeric_limits<double>::infinity();
}

template <>
constexpr float lowest() {
  return -std::numeric_limits<float>::infinity();
}

template <typename T>
constexpr T highest() {
  return std::numeric_limits<T>::max();
}

template <>
constexpr double highest() {
  return std::numeric_limits<double>::infinity();
}

template <>
constexpr float highest() {
  return std::numeric_limits<float>::infinity();
}

template <std::size_t I, class T, std::size_t... N>
decltype(auto) tuple_slice_impl(T&& t, mp11::index_sequence<N...>) {
  return std::forward_as_tuple(std::get<(N + I)>(std::forward<T>(t))...);
}

template <std::size_t I, std::size_t N, class T>
decltype(auto) tuple_slice(T&& t) {
  static_assert(I + N <= mp11::mp_size<remove_cvref_t<T>>::value,
                "I and N must describe a slice");
  return tuple_slice_impl<I>(std::forward<T>(t), mp11::make_index_sequence<N>{});
}

#define BOOST_HISTOGRAM_DETECT(name, cond)   \
  template <class T, class = decltype(cond)> \
  struct name##_impl {};                     \
  template <class T>                         \
  struct name : mp11::mp_valid<name##_impl, T>::type {}

#define BOOST_HISTOGRAM_DETECT_BINARY(name, cond)     \
  template <class T, class U, class = decltype(cond)> \
  struct name##_impl {};                              \
  template <class T, class U = T>                     \
  struct name : mp11::mp_valid<name##_impl, T, U>::type {}

BOOST_HISTOGRAM_DETECT(has_method_metadata, (std::declval<T&>().metadata()));

// resize has two overloads, trying to get pmf in this case always fails
BOOST_HISTOGRAM_DETECT(has_method_resize, (std::declval<T&>().resize(0)));

BOOST_HISTOGRAM_DETECT(has_method_size, &T::size);

BOOST_HISTOGRAM_DETECT(has_method_clear, &T::clear);

BOOST_HISTOGRAM_DETECT(has_method_lower, &T::lower);

BOOST_HISTOGRAM_DETECT(has_method_value, &T::value);

BOOST_HISTOGRAM_DETECT(has_method_update, (&T::update));

BOOST_HISTOGRAM_DETECT(has_method_reset, (std::declval<T>().reset(0)));

template <typename T>
using get_value_method_return_type_impl = decltype(std::declval<T&>().value(0));

template <typename T, typename R>
using has_method_value_with_convertible_return_type = typename std::is_convertible<
    mp11::mp_eval_or<void, get_value_method_return_type_impl, T>, R>::type;

BOOST_HISTOGRAM_DETECT(has_method_options, (&T::options));

BOOST_HISTOGRAM_DETECT(has_allocator, &T::get_allocator);

BOOST_HISTOGRAM_DETECT(is_indexable, (std::declval<T&>()[0]));

BOOST_HISTOGRAM_DETECT(is_transform, (&T::forward, &T::inverse));

BOOST_HISTOGRAM_DETECT(is_indexable_container,
                       (std::declval<T>()[0], &T::size, std::begin(std::declval<T>()),
                        std::end(std::declval<T>())));

BOOST_HISTOGRAM_DETECT(is_vector_like,
                       (std::declval<T>()[0], &T::size, std::declval<T>().resize(0),
                        std::begin(std::declval<T>()), std::end(std::declval<T>())));

BOOST_HISTOGRAM_DETECT(is_array_like,
                       (std::declval<T>()[0], &T::size, std::tuple_size<T>::value,
                        std::begin(std::declval<T>()), std::end(std::declval<T>())));

BOOST_HISTOGRAM_DETECT(is_map_like,
                       (std::declval<typename T::key_type>(),
                        std::declval<typename T::mapped_type>(),
                        std::begin(std::declval<T>()), std::end(std::declval<T>())));

// ok: is_axis is false for axis::variant, operator() is templated
BOOST_HISTOGRAM_DETECT(is_axis, (&T::size, &T::index));

BOOST_HISTOGRAM_DETECT(is_iterable,
                       (std::begin(std::declval<T&>()), std::end(std::declval<T&>())));

BOOST_HISTOGRAM_DETECT(is_iterator,
                       (typename std::iterator_traits<T>::iterator_category()));

BOOST_HISTOGRAM_DETECT(is_streamable,
                       (std::declval<std::ostream&>() << std::declval<T&>()));

BOOST_HISTOGRAM_DETECT(has_operator_preincrement, (++std::declval<T&>()));

BOOST_HISTOGRAM_DETECT_BINARY(has_operator_equal,
                              (std::declval<const T&>() == std::declval<const U&>()));

BOOST_HISTOGRAM_DETECT_BINARY(has_operator_radd,
                              (std::declval<T&>() += std::declval<U&>()));

BOOST_HISTOGRAM_DETECT_BINARY(has_operator_rsub,
                              (std::declval<T&>() -= std::declval<U&>()));

BOOST_HISTOGRAM_DETECT_BINARY(has_operator_rmul,
                              (std::declval<T&>() *= std::declval<U&>()));

BOOST_HISTOGRAM_DETECT_BINARY(has_operator_rdiv,
                              (std::declval<T&>() /= std::declval<U&>()));

BOOST_HISTOGRAM_DETECT(has_threading_support, (T::has_threading_support));

template <typename T>
using is_storage = mp11::mp_and<is_indexable_container<T>, has_method_reset<T>,
                                has_threading_support<T>>;

template <class T>
using is_adaptible = mp11::mp_or<is_vector_like<T>, is_array_like<T>, is_map_like<T>>;

template <class T, class _ = remove_cvref_t<T>,
          class = std::enable_if_t<(is_storage<_>::value || is_adaptible<_>::value)>>
struct requires_storage_or_adaptible {};

template <typename T>
struct is_tuple_impl : std::false_type {};

template <typename... Ts>
struct is_tuple_impl<std::tuple<Ts...>> : std::true_type {};

template <typename T>
using is_tuple = typename is_tuple_impl<T>::type;

template <typename T>
struct is_axis_variant_impl : std::false_type {};

template <typename... Ts>
struct is_axis_variant_impl<axis::variant<Ts...>> : std::true_type {};

template <typename T>
using is_axis_variant = typename is_axis_variant_impl<T>::type;

template <typename T>
using is_any_axis = mp11::mp_or<is_axis<T>, is_axis_variant<T>>;

template <typename T>
using is_sequence_of_axis = mp11::mp_and<is_iterable<T>, is_axis<mp11::mp_first<T>>>;

template <typename T>
using is_sequence_of_axis_variant =
    mp11::mp_and<is_iterable<T>, is_axis_variant<mp11::mp_first<T>>>;

template <typename T>
using is_sequence_of_any_axis =
    mp11::mp_and<is_iterable<T>, is_any_axis<mp11::mp_first<T>>>;

template <typename T>
struct is_weight_impl : std::false_type {};

template <typename T>
struct is_weight_impl<weight_type<T>> : std::true_type {};

template <typename T>
using is_weight = is_weight_impl<remove_cvref_t<T>>;

template <typename T>
struct is_sample_impl : std::false_type {};

template <typename T>
struct is_sample_impl<sample_type<T>> : std::true_type {};

template <typename T>
using is_sample = is_sample_impl<remove_cvref_t<T>>;

// poor-mans concept checks
template <class T, class = std::enable_if_t<is_iterator<remove_cvref_t<T>>::value>>
struct requires_iterator {};

template <class T, class = std::enable_if_t<is_iterable<remove_cvref_t<T>>::value>>
struct requires_iterable {};

template <class T, class = std::enable_if_t<is_axis<remove_cvref_t<T>>::value>>
struct requires_axis {};

template <class T, class = std::enable_if_t<is_any_axis<remove_cvref_t<T>>::value>>
struct requires_any_axis {};

template <class T,
          class = std::enable_if_t<is_sequence_of_axis<remove_cvref_t<T>>::value>>
struct requires_sequence_of_axis {};

template <class T,
          class = std::enable_if_t<is_sequence_of_axis_variant<remove_cvref_t<T>>::value>>
struct requires_sequence_of_axis_variant {};

template <class T,
          class = std::enable_if_t<is_sequence_of_any_axis<remove_cvref_t<T>>::value>>
struct requires_sequence_of_any_axis {};

template <class T,
          class = std::enable_if_t<is_any_axis<mp11::mp_first<remove_cvref_t<T>>>::value>>
struct requires_axes {};

template <class T, class U, class = std::enable_if_t<std::is_convertible<T, U>::value>>
struct requires_convertible {};

template <class T>
auto make_default(const T& t) {
  return static_if<has_allocator<T>>([](const auto& t) { return T(t.get_allocator()); },
                                     [](const auto&) { return T{}; }, t);
}

template <class T>
constexpr bool relaxed_equal(const T& a, const T& b) noexcept {
  return static_if<has_operator_equal<T>>(
      [](const auto& a, const auto& b) { return a == b; },
      [](const auto&, const auto&) { return true; }, a, b);
}

template <class T>
using get_scale_type_helper = typename T::value_type;

template <class T>
using get_scale_type = mp11::mp_eval_or<T, detail::get_scale_type_helper, T>;

struct one_unit {};

template <class T>
T operator*(T&& t, const one_unit&) {
  return std::forward<T>(t);
}

template <class T>
T operator/(T&& t, const one_unit&) {
  return std::forward<T>(t);
}

template <class T>
using get_unit_type_helper = typename T::unit_type;

template <class T>
using get_unit_type = mp11::mp_eval_or<one_unit, detail::get_unit_type_helper, T>;

template <class T, class R = get_scale_type<T>>
R get_scale(const T& t) {
  return t / get_unit_type<T>();
}

template <class T, class Default>
using replace_default = mp11::mp_if<std::is_same<T, use_default>, Default, T>;

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
