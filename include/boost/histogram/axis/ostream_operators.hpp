// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// String representations here evaluate correctly in Python.

#ifndef BOOST_HISTOGRAM_AXIS_OSTREAM_OPERATORS_HPP
#define BOOST_HISTOGRAM_AXIS_OSTREAM_OPERATORS_HPP

#include <boost/core/typeinfo.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/polymorphic_bin.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/throw_exception.hpp>
#include <iomanip>
#include <ostream>
#include <type_traits>

namespace boost {
namespace histogram {

namespace detail {
template <typename T>
const char* to_string(const axis::transform::identity<T>&) {
  return "";
}
template <typename T>
const char* to_string(const axis::transform::log<T>&) {
  return "_log";
}
template <typename T>
const char* to_string(const axis::transform::sqrt<T>&) {
  return "_sqrt";
}
template <typename T>
const char* to_string(const axis::transform::pow<T>&) {
  return "_pow";
}

template <typename OStream, typename T>
void stream_metadata(OStream& os, const T& t) {
  detail::static_if<detail::is_streamable<T>>(
      [&os](const auto& t) {
        std::ostringstream oss;
        oss << t;
        if (!oss.str().empty()) { os << ", metadata=" << std::quoted(oss.str()); }
      },
      [&os](const auto&) {
        using U = detail::unqual<T>;
        os << ", metadata=" << boost::core::demangled_name(BOOST_CORE_TYPEID(U));
      },
      t);
}

template <typename OStream>
void stream_options(OStream& os, const axis::option_type o) {
  os << ", options=" << o;
}

template <typename OStream, typename T>
void stream_transform(OStream&, const T&) {}

template <typename OStream, typename T>
void stream_transform(OStream& os, const axis::transform::pow<T>& t) {
  os << ", power=" << t.power;
}

template <typename OStream, typename T>
void stream_value(OStream& os, const T& t) {
  os << t;
}

template <typename OStream, typename... Ts>
void stream_value(OStream& os, const std::basic_string<Ts...>& t) {
  os << std::quoted(t);
}

} // namespace detail

namespace axis {

template <typename C, typename T>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     const axis::option_type o) {
  using opt = axis::option_type;
  if (o == opt::none)
    os << "none";
  else {
    bool first = true;
    for (auto x : {opt::underflow, opt::overflow, opt::circular}) {
      if (!(o & x)) continue;
      if (first)
        first = false;
      else
        os << " | ";
      switch (x) {
        case axis::option_type::underflow: os << "underflow"; break;
        case axis::option_type::overflow: os << "overflow"; break;
        case axis::option_type::circular: os << "circular"; break;
        default: break;
      }
    }
  }
  return os;
}

template <typename C, typename T>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os, const null_type&) {
  return os; // do nothing
}

template <typename C, typename T, typename U>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     const interval_view<U>& i) {
  os << "[" << i.lower() << ", " << i.upper() << ")";
  return os;
}

template <typename C, typename T, typename U>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     const polymorphic_bin<U>& i) {
  if (i.is_discrete())
    os << i;
  else
    os << "[" << i.lower() << ", " << i.upper() << ")";
  return os;
}

template <typename C, typename T, typename Tr, typename M, option_type O>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     const regular<Tr, M, O>& a) {
  os << "regular" << detail::to_string(a.transform()) << "(" << a.size() << ", "
     << a.value(0) << ", " << a.value(a.size());
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  detail::stream_transform(os, a.transform());
  os << ")";
  return os;
}

template <typename C, typename T, typename U, typename M, option_type O>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     const integer<U, M, O>& a) {
  os << "integer(" << a.value(0) << ", " << a.value(a.size());
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  os << ")";
  return os;
}

template <typename C, typename T, typename U, typename M, option_type O, typename A>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     const variable<U, M, O, A>& a) {
  os << "variable(" << a.value(0);
  for (int i = 1, n = a.size(); i <= n; ++i) { os << ", " << a.value(i); }
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  os << ")";
  return os;
}

template <typename C, typename T, typename U, typename M, option_type O, typename A>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     const category<U, M, O, A>& a) {
  os << "category(";
  for (int i = 0, n = a.size(); i < n; ++i) {
    detail::stream_value(os, a.value(i));
    os << (i == (a.size() - 1) ? "" : ", ");
  }
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  os << ")";
  return os;
}

template <typename C, typename T, typename... Ts>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                     const variant<Ts...>& v) {
  visit(
      [&os](const auto& x) {
        using A = detail::unqual<decltype(x)>;
        detail::static_if<detail::is_streamable<A>>(
            [&os](const auto& x) { os << x; },
            [](const auto&) {
              BOOST_THROW_EXCEPTION(std::runtime_error(
                  detail::cat(boost::core::demangled_name(BOOST_CORE_TYPEID(A)),
                              " is not streamable")));
            },
            x);
      },
      v);
  return os;
}

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
