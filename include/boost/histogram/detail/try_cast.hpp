// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_TRY_CAST_HPP
#define BOOST_HISTOGRAM_DETAIL_TRY_CAST_HPP

#include <boost/core/demangle.hpp>
#include <boost/core/typeinfo.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {
template <class T, class E, class U>
T try_cast_impl(std::false_type, U&&) {
  BOOST_THROW_EXCEPTION(
      E(detail::cat("cannot cast ", core::demangled_name(BOOST_CORE_TYPEID(T)), " to ",
                    core::demangled_name(BOOST_CORE_TYPEID(U)))));
}

template <class T, class E, class U>
T try_cast_impl(std::true_type, U&& u) {
  return static_cast<T>(u);
}

// cast fails at runtime with exception E instead of compile-time
template <class T, class E, class U>
T try_cast(U&& u) {
  return try_cast_impl<T, E>(std::is_convertible<U, T>{}, std::forward<U>(u));
}
} // namespace detail
} // namespace histogram
} // namespace boost

#endif
