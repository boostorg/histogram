// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_OSTREAM_OPERATORS_HPP_
#define _BOOST_HISTOGRAM_AXIS_OSTREAM_OPERATORS_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/math/constants/constants.hpp>
#include <ostream>

namespace boost {
namespace histogram {
namespace axis {

template <typename RealType>
inline std::ostream &operator<<(std::ostream &os, const regular<RealType> &a) {
  os << "regular(" << a.size() << ", " << a[0].lower() << ", " << a[a.size()].lower();
  if (!a.label().empty()) {
    os << ", label=";
    ::boost::histogram::detail::escape(os, a.label());
  }
  if (!a.uoflow()) {
    os << ", uoflow=False";
  }
  os << ")";
  return os;
}

template <typename RealType>
inline std::ostream &operator<<(std::ostream &os, const circular<RealType> &a) {
  os << "circular(" << a.size();
  if (a.phase() != 0.0) {
    os << ", phase=" << a.phase();
  }
  if (a.perimeter() != RealType(math::double_constants::two_pi)) {
    os << ", perimeter=" << a.perimeter();
  }
  if (!a.label().empty()) {
    os << ", label=";
    ::boost::histogram::detail::escape(os, a.label());
  }
  os << ")";
  return os;
}

template <typename RealType>
inline std::ostream &operator<<(std::ostream &os, const variable<RealType> &a) {
  os << "variable(" << a[0].lower();
  for (int i = 1; i <= a.size(); ++i) {
    os << ", " << a[i].lower();
  }
  if (!a.label().empty()) {
    os << ", label=";
    ::boost::histogram::detail::escape(os, a.label());
  }
  if (!a.uoflow()) {
    os << ", uoflow=False";
  }
  os << ")";
  return os;
}

template <typename IntType>
inline std::ostream &operator<<(std::ostream &os, const integer<IntType> &a) {
  os << "integer(" << a[0].lower() << ", " << a[a.size()].lower();
  if (!a.label().empty()) {
    os << ", label=";
    ::boost::histogram::detail::escape(os, a.label());
  }
  if (!a.uoflow()) {
    os << ", uoflow=False";
  }
  os << ")";
  return os;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const category<T> &a) {
  os << "category(";
  for (int i = 0; i < a.size(); ++i) {
    os << a[i] << (i == (a.size() - 1) ? "" : ", ");
  }
  if (!a.label().empty()) {
    os << ", label=";
    ::boost::histogram::detail::escape(os, a.label());
  }
  os << ")";
  return os;
}

template <>
inline std::ostream &operator<<(std::ostream &os, const category<std::string> &a) {
  os << "category(";
  for (int i = 0; i < a.size(); ++i) {
    ::boost::histogram::detail::escape(os, a[i]);
    os << (i == (a.size() - 1) ? "" : ", ");
  }
  if (!a.label().empty()) {
    os << ", label=";
    ::boost::histogram::detail::escape(os, a.label());
  }
  os << ")";
  return os;
}

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
