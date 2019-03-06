// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_TRAITS_HPP
#define BOOST_HISTOGRAM_AXIS_TRAITS_HPP

#include <boost/core/typeinfo.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <utility>

namespace boost {
namespace histogram {
namespace detail {

template <class T>
using static_options_impl = axis::option::bitset<T::options()>;

template <class FIntArg, class FDoubleArg, class T>
decltype(auto) value_method_switch(FIntArg&& iarg, FDoubleArg&& darg, const T& t) {
  return static_if<has_method_value<T>>(
      [](FIntArg&& iarg, FDoubleArg&& darg, const auto& t) {
        using A = remove_cvref_t<decltype(t)>;
        return static_if<std::is_same<arg_type<decltype(&A::value), 0>, int>>(
            std::forward<FIntArg>(iarg), std::forward<FDoubleArg>(darg), t);
      },
      [](FIntArg&&, FDoubleArg&&, const auto& t) {
        using A = remove_cvref_t<decltype(t)>;
        BOOST_THROW_EXCEPTION(std::runtime_error(detail::cat(
            boost::core::demangled_name(BOOST_CORE_TYPEID(A)), " has no value method")));
        return 0;
      },
      std::forward<FIntArg>(iarg), std::forward<FDoubleArg>(darg), t);
}

template <class R1, class R2, class FIntArg, class FDoubleArg, class T>
R2 value_method_switch_with_return_type(FIntArg&& iarg, FDoubleArg&& darg, const T& t) {
  return static_if<has_method_value_with_convertible_return_type<T, R1>>(
      [](FIntArg&& iarg, FDoubleArg&& darg, const auto& t) -> R2 {
        using A = remove_cvref_t<decltype(t)>;
        return static_if<std::is_same<arg_type<decltype(&A::value), 0>, int>>(
            std::forward<FIntArg>(iarg), std::forward<FDoubleArg>(darg), t);
      },
      [](FIntArg&&, FDoubleArg&&, const auto&) -> R2 {
        BOOST_THROW_EXCEPTION(std::runtime_error(
            detail::cat(boost::core::demangled_name(BOOST_CORE_TYPEID(T)),
                        " has no value method or return type is not convertible to ",
                        boost::core::demangled_name(BOOST_CORE_TYPEID(R1)))));
      },
      std::forward<FIntArg>(iarg), std::forward<FDoubleArg>(darg), t);
}
} // namespace detail

namespace axis {
namespace traits {
template <class T>
decltype(auto) metadata(T&& t) noexcept {
  return detail::static_if<detail::has_method_metadata<detail::remove_cvref_t<T>>>(
      [](auto&& x) -> decltype(auto) { return x.metadata(); },
      [](T &&) -> detail::copy_qualifiers<T, axis::null_type> {
        static axis::null_type m;
        return m;
      },
      std::forward<T>(t));
}

template <class T>
using static_options =
    detail::mp_eval_or<mp11::mp_if<detail::has_method_update<detail::remove_cvref_t<T>>,
                                   axis::option::growth_t, axis::option::none_t>,
                       detail::static_options_impl, detail::remove_cvref_t<T>>;

template <class T>
constexpr unsigned options(const T& t) noexcept {
  // cannot reuse static_options here, because this should also work for axis::variant
  return detail::static_if<detail::has_method_options<T>>(
      [](const auto& t) { return t.options(); },
      [](const auto&) {
        return detail::has_method_update<T>::value ? axis::option::growth_t::value
                                                   : axis::option::none_t::value;
      },
      t);
}

template <class T>
constexpr axis::index_type extend(const T& t) noexcept {
  const auto opt = options(t);
  return t.size() + static_cast<bool>(opt & option::underflow_t::value) +
         static_cast<bool>(opt & option::overflow_t::value);
}

template <class T>
decltype(auto) value(const T& axis, double idx) {
  return detail::value_method_switch(
      [idx](const auto& a) { return a.value(static_cast<int>(idx)); },
      [idx](const auto& a) { return a.value(idx); }, axis);
}

template <class... Ts, class U>
auto index(const axis::variant<Ts...>& axis, const U& value) {
  return axis.index(value);
}

template <class T, class U>
auto index(const T& axis, const U& value) {
  using V = detail::arg_type<decltype(&T::index)>;
  return detail::static_if<std::is_convertible<U, V>>(
      [&value](const auto& axis) {
        using A = detail::remove_cvref_t<decltype(axis)>;
        using V2 = detail::arg_type<decltype(&A::index)>;
        return axis.index(static_cast<V2>(value));
      },
      [](const T&) {
        BOOST_THROW_EXCEPTION(std::invalid_argument(
            detail::cat(boost::core::demangled_name(BOOST_CORE_TYPEID(T)),
                        ": cannot convert argument of type ",
                        boost::core::demangled_name(BOOST_CORE_TYPEID(U)), " to ",
                        boost::core::demangled_name(BOOST_CORE_TYPEID(V)))));
        return 0;
      },
      axis);
}

template <class... Ts, class U>
auto update(axis::variant<Ts...>& t, const U& u) {
  return t.update(u);
}

template <class T, class U>
std::pair<int, int> update(T& axis, const U& value) {
  using V = detail::arg_type<decltype(&T::index)>;
  return detail::static_if<std::is_convertible<U, V>>(
      [&value](auto& a) {
        return detail::static_if<
            detail::has_method_update<detail::remove_cvref_t<decltype(a)>>>(
            [&value](auto& a) { return a.update(value); },
            [&value](const auto& a) { return std::make_pair(a.index(value), 0); }, a);
      },
      [](T&) {
        BOOST_THROW_EXCEPTION(std::invalid_argument(
            detail::cat(boost::core::demangled_name(BOOST_CORE_TYPEID(T)),
                        ": cannot convert argument of type ",
                        boost::core::demangled_name(BOOST_CORE_TYPEID(U)), " to ",
                        boost::core::demangled_name(BOOST_CORE_TYPEID(V)))));
        return std::make_pair(0, 0);
      },
      axis);
}

template <class R, class T>
R value_as(const T& t, double idx) {
  return detail::value_method_switch_with_return_type<R, R>(
      [idx](const auto& a) -> R { return a.value(static_cast<int>(idx)); },
      [idx](const auto& a) -> R { return a.value(idx); }, t);
}

template <class T>
decltype(auto) width(const T& t, int idx) {
  return detail::value_method_switch(
      [](const auto&) { return 0; },
      [idx](const auto& a) { return a.value(idx + 1) - a.value(idx); }, t);
}

template <class R, class T>
R width_as(const T& t, int idx) {
  return detail::value_method_switch_with_return_type<R, R>(
      [](const auto&) { return R(); },
      [idx](const auto& a) -> R { return a.value(idx + 1) - a.value(idx); }, t);
}

} // namespace traits
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
