// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_TRAITS_HPP
#define BOOST_HISTOGRAM_AXIS_TRAITS_HPP

#include <boost/core/typeinfo.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <utility>

namespace boost {
namespace histogram {
namespace detail {

template <class FIntArg, class FDoubleArg, class T>
decltype(auto) value_method_switch(FIntArg&& iarg, FDoubleArg&& darg, const T& t) {
  return static_if<has_method_value<T>>(
      [](FIntArg&& iarg, FDoubleArg&& darg, const auto& t) {
        using A = naked<decltype(t)>;
        return static_if<std::is_same<arg_type<decltype(&A::value), 0>, int>>(
            std::forward<FIntArg>(iarg), std::forward<FDoubleArg>(darg), t);
      },
      [](FIntArg&&, FDoubleArg&&, const auto& t) {
        using A = naked<decltype(t)>;
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
        using A = naked<decltype(t)>;
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
  return detail::static_if<detail::has_method_metadata<detail::naked<T>>>(
      [](auto&& x) -> decltype(auto) { return x.metadata(); },
      [](T &&) -> detail::copy_qualifiers<T, axis::null_type> {
        static axis::null_type m;
        return m;
      },
      std::forward<T>(t));
}

template <class T>
option_type options(const T& axis) noexcept {
  return detail::static_if<detail::has_method_options<T>>(
      [](const auto& a) { return a.options(); },
      [](const T&) { return axis::option_type::none; }, axis);
}

template <class T>
int extend(const T& t) noexcept {
  const auto opt = options(t);
  return t.size() + (opt & option_type::underflow) + (opt & option_type::overflow);
}

template <class T>
decltype(auto) value(const T& axis, double idx) {
  return detail::value_method_switch(
      [idx](const auto& a) { return a.value(static_cast<int>(idx)); },
      [idx](const auto& a) { return a.value(idx); }, axis);
}

template <class... Ts, class U>
auto index(const axis::variant<Ts...>& axis, const U& value) {
  return axis(value);
}

template <class T, class U>
auto index(const T& axis, const U& value) {
  return detail::static_if<std::is_convertible<U, detail::arg_type<T>>>(
      [&value](const auto& axis) { return axis(value); },
      [](const T&) {
        using V = detail::arg_type<T>;
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
  return detail::static_if<std::is_convertible<U, detail::arg_type<T>>>(
      [&value](auto& a) {
        return detail::static_if<detail::has_method_update<detail::naked<decltype(a)>>>(
            [&value](auto& a) { return a.update(value); },
            [&value](const auto& a) { return std::make_pair(a(value), 0); }, a);
      },
      [](T&) {
        using V = detail::arg_type<T>;
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
