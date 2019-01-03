// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_OSTREAM_OPERATORS_HPP
#define BOOST_HISTOGRAM_OSTREAM_OPERATORS_HPP

#include <boost/histogram/fwd.hpp>
#include <ostream>

namespace boost {
namespace histogram {

namespace detail {
template <class OStream, class T>
void grid_ostream_impl(OStream& os, const char* prefix, const T& t) {
  os << prefix << "(";
  t.for_each_axis([&](const auto& a) { os << "\n  " << a << ","; });
  os << (t.rank() ? "\n)" : ")");
}
} // namespace detail

template <typename CharT, typename Traits, typename A, typename S>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const grid<A, S>& h) {
  detail::grid_ostream_impl(os, "grid", h);
  return os;
}

template <typename CharT, typename Traits, typename A, typename S>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const histogram<A, S>& h) {
  detail::grid_ostream_impl(os, "histogram", h);
  return os;
}

} // namespace histogram
} // namespace boost

#endif
