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
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/throw_exception.hpp>
#include <utility>

namespace boost {
namespace histogram {
namespace detail {
template <typename T>
constexpr axis::option_type options_impl(const T&, std::false_type) {
  return axis::option_type::none;
}

template <typename T>
axis::option_type options_impl(const T& t, std::true_type) {
  return t.options();
}

template <typename FIntArg, typename FDoubleArg, typename T>
decltype(auto) value_method_switch(FIntArg&& iarg, FDoubleArg&& darg, const T& t) {
  using U = unqual<T>;
  return static_if<has_method_value<U>>(
      [](FIntArg&& iarg, FDoubleArg&& darg, const auto& t) {
        using A = unqual<decltype(t)>;
        return static_if<std::is_same<arg_type<decltype(&A::value), 0>, int>>(
            std::forward<FIntArg>(iarg), std::forward<FDoubleArg>(darg), t);
      },
      [](FIntArg&&, FDoubleArg&&, const auto& t) {
        using A = unqual<decltype(t)>;
        BOOST_THROW_EXCEPTION(std::runtime_error(detail::cat(
            boost::core::demangled_name(BOOST_CORE_TYPEID(A)), " has no value method")));
        return 0;
      },
      std::forward<FIntArg>(iarg), std::forward<FDoubleArg>(darg), t);
}

template <typename R1, typename R2, typename FIntArg, typename FDoubleArg, typename T>
R2 value_method_switch_with_return_type(FIntArg&& iarg, FDoubleArg&& darg, const T& t) {
  using U = unqual<T>;
  return static_if<has_method_value_with_convertible_return_type<U, R1>>(
      [](FIntArg&& iarg, FDoubleArg&& darg, const auto& t) -> R2 {
        using A = unqual<decltype(t)>;
        return static_if<std::is_same<arg_type<decltype(&A::value), 0>, int>>(
            std::forward<FIntArg>(iarg), std::forward<FDoubleArg>(darg), t);
      },
      [](FIntArg&&, FDoubleArg&&, const auto&) -> R2 {
        BOOST_THROW_EXCEPTION(std::runtime_error(
            detail::cat(boost::core::demangled_name(BOOST_CORE_TYPEID(U)),
                        " has no value method or return type is not convertible to ",
                        boost::core::demangled_name(BOOST_CORE_TYPEID(R1)))));
      },
      std::forward<FIntArg>(iarg), std::forward<FDoubleArg>(darg), t);
}
} // namespace detail

namespace axis {
namespace traits {
template <typename T>
decltype(auto) metadata(T&& t) noexcept {
  return detail::static_if<detail::has_method_metadata<T>>(
      [](auto&& x) -> decltype(auto) { return x.metadata(); },
      [](T &&) -> detail::copy_qualifiers<T, axis::null_type> {
        static axis::null_type m;
        return m;
      },
      std::forward<T>(t));
}

template <typename T>
option_type options(const T& t) noexcept {
  return detail::options_impl(t, detail::has_method_options<T>());
}

template <typename T>
int extend(const T& t) noexcept {
  const auto opt = options(t);
  return t.size() + (opt & option_type::underflow) + (opt & option_type::overflow);
}

template <typename T>
decltype(auto) value(const T& t, double idx) {
  return detail::value_method_switch(
      [idx](const auto& a) { return a.value(static_cast<int>(idx)); },
      [idx](const auto& a) { return a.value(idx); }, t);
}

template <typename R, typename T>
R value_as(const T& t, double idx) {
  return detail::value_method_switch_with_return_type<R, R>(
      [idx](const auto& a) -> R { return a.value(static_cast<int>(idx)); },
      [idx](const auto& a) -> R { return a.value(idx); }, t);
}

template <typename T>
decltype(auto) width(const T& t, int idx) {
  return detail::value_method_switch(
      [](const auto&) { return 0; },
      [idx](const auto& a) { return a.value(idx + 1) - a.value(idx); }, t);
}

template <typename R, typename T>
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
