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
#include <boost/utility/string_view.hpp>
#include <ostream>

namespace boost {
namespace histogram {
namespace axis {

namespace detail {
inline ::boost::string_view to_string(const transform::identity&) { return {}; }
inline ::boost::string_view to_string(const transform::log&) { return {"_log", 4}; }
inline ::boost::string_view to_string(const transform::sqrt&) { return {"_sqrt", 5}; }

template <typename OStream>
void escape_string(OStream& os, const ::boost::string_view s) {
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
} // namespace detail

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
  // if (!a.label().empty()) {
  //   os << ", label=";
  //   detail::escape_string(os, a.label());
  // }
  // if (!a.uoflow()) { os << ", uoflow=False"; }
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename T, typename M>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os, const regular<axis::transform::pow<T>, M>& a) {
  os << "regular_pow(" << a.size() << ", " << a.lower(0) << ", " << a.lower(a.size())
     << ", " << a.transform().power;
  // if (!a.label().empty()) {
  //   os << ", label=";
  //   detail::escape_string(os, a.label());
  // }
  // if (!a.uoflow()) { os << ", uoflow=False"; }
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename T, typename A>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const circular<T, A>& a) {
  os << "circular(" << a.size();
  const auto phase = a.lower(0);
  const auto perimeter = a.lower(a.size()) - a.lower(0);
  if (phase != 0.0) { os << ", phase=" << phase; }
  if (perimeter != circular<T, A>::two_pi()) {
    os << ", perimeter=" << perimeter;
  }
  // if (!a.label().empty()) {
  //   os << ", label=";
  //   detail::escape_string(os, a.label());
  // }
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename T, typename A>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const variable<T, A>& a) {
  os << "variable(" << a.lower(0);
  for (int i = 1; i <= a.size(); ++i) { os << ", " << a.lower(i); }
  // if (!a.label().empty()) {
  //   os << ", label=";
  //   detail::escape_string(os, a.label());
  // }
  // if (!a.uoflow()) { os << ", uoflow=False"; }
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename T, typename A>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const integer<T, A>& a) {
  os << "integer(" << a.lower(0) << ", " << a.lower(a.size());
  // if (!a.label().empty()) {
  //   os << ", label=";
  //   detail::escape_string(os, a.label());
  // }
  // if (!a.uoflow()) { os << ", uoflow=False"; }
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename T, typename A>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const category<T, A>& a) {
  os << "category(";
  for (int i = 0; i < a.size(); ++i) { os << a[i] << (i == (a.size() - 1) ? "" : ", "); }
  // if (!a.label().empty()) {
  //   os << ", label=";
  //   detail::escape_string(os, a.label());
  // }
  os << ")";
  return os;
}

template <typename CharT, typename Traits, typename A>
inline std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os, const category<std::string, A>& a) {
  os << "category(";
  for (int i = 0; i < a.size(); ++i) {
    detail::escape_string(os, a.value(i));
    os << (i == (a.size() - 1) ? "" : ", ");
  }
  // if (!a.label().empty()) {
  //   os << ", label=";
  //   detail::escape_string(os, a.label());
  // }
  os << ")";
  return os;
}

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
