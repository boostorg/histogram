// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// String representations here evaluate correctly in Python.

#ifndef BOOST_HISTOGRAM_AXIS_OSTREAM_OPERATORS_HPP
#define BOOST_HISTOGRAM_AXIS_OSTREAM_OPERATORS_HPP

#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/types.hpp>
#include <boost/histogram/axis/value_view.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <ostream>

namespace boost {
namespace histogram {

namespace detail {
template <typename T> const char* to_string(const axis::transform::identity<T>&) { return ""; }
template <typename T> const char* to_string(const axis::transform::log<T>&) { return "_log"; }
template <typename T> const char* to_string(const axis::transform::sqrt<T>&) { return "_sqrt"; }
template <typename T> const char* to_string(const axis::transform::quantity<T>&) { return "_quantity"; }
template <typename T> const char* to_string(const axis::transform::pow<T>&) { return "_pow"; }

template <typename OStream>
void escape_string(OStream& os, const std::string& s) {
  os << '\'';
  for (auto sit = s.begin(); sit != s.end(); ++sit) {
    if (*sit == '\'' && (sit == s.begin() || *(sit - 1) != '\\')) {
      os << "\\\'";
    } else {
      os << *sit;
    }
  }
  os << '\'';
}

template <typename OStream, typename T>
void stream_metadata(OStream& os, const T& t) {
  if (detail::is_streamable<T>::value) {
    std::ostringstream oss;
    oss << t;
    if (!oss.str().empty())
      os << ", metadata=";
      escape_string(os, oss.str());
  }
}

template <typename OStream>
void stream_options(OStream& os, const axis::option_type o) {
  os << ", options=";
  switch (o) {
    case axis::option_type::none: os << "none"; break;
    case axis::option_type::overflow: os << "overflow"; break;
    case axis::option_type::underflow_and_overflow: os << "underflow_and_overflow"; break;
  }
}

template <typename OStream, typename T>
void stream_transform(OStream&, const T&) {}

template <typename OStream, typename T>
void stream_transform(OStream& os, const axis::transform::pow<T>& t) {
  os << ", power=" << t.power;
}

template <typename OStream, typename T>
void stream_transform(OStream& os, const axis::transform::quantity<T>& t) {
  os << ", unit=" << t.unit;
}

} // namespace detail

namespace axis {

template <typename CharT, typename Traits, typename T>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const interval_view<T>& i) {
  os << "[" << i.lower() << ", " << i.upper() << ")";
  return os;
}

template <typename CharT, typename Traits, typename T>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const value_view<T>& i) {
  os << i.value();
  return os;
}

template <typename CharT, typename Traits, typename T, typename M>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const regular<T, M>& a) {
  os << "regular" << detail::to_string(a.transform()) << "(" << a.size() << ", "
     << a.lower(0) << ", " << a.lower(a.size());
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  detail::stream_transform(os, a.transform());
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename T, typename A>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const circular<T, A>& a) {
  os << "circular(" << a.size()
     << ", " << a.lower(0) << ", " << a.lower(a.size());
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename T, typename A>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const variable<T, A>& a) {
  os << "variable(" << a.lower(0);
  for (unsigned i = 1; i <= a.size(); ++i) { os << ", " << a.lower(i); }
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename T, typename A>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const integer<T, A>& a) {
  os << "integer(" << a.lower(0) << ", " << a.lower(a.size());
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename T, typename A>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const category<T, A>& a) {
  os << "category(";
  for (unsigned i = 0; i < a.size(); ++i) { os << a[i] << (i == (a.size() - 1) ? "" : ", "); }
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename A>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os, const category<std::string, A>& a) {
  os << "category(";
  for (unsigned i = 0; i < a.size(); ++i) {
    detail::escape_string(os, a.value(i));
    os << (i == (a.size() - 1) ? "" : ", ");
  }
  detail::stream_metadata(os, a.metadata());
  detail::stream_options(os, a.options());
  os << ")";
  return os;
}

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
