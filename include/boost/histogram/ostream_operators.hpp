// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_OSTREAM_OPERATORS_HPP_
#define _BOOST_HISTOGRAM_OSTREAM_OPERATORS_HPP_

#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <ostream>

namespace boost {
namespace histogram {

namespace detail {
template <typename OStream>
struct axis_ostream_visitor {
  OStream& os_;
  explicit axis_ostream_visitor(OStream& os) : os_(os) {}
  template <typename Axis>
  void operator()(const Axis& a) const {
    os_ << "\n  " << a << ",";
  }
};
} // namespace detail

template <typename CharT, typename Traits, typename A, typename S>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const histogram<A, S>& h) {
  using OS = std::basic_ostream<CharT, Traits>;
  os << "histogram(";
  h.for_each_axis(detail::axis_ostream_visitor<OS>(os));
  os << (h.dim() ? "\n)" : ")");
  return os;
}

template <typename CharT, typename Traits, typename W>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const weight_counter<W>& x) {
  os << "weight_counter(" << x.value() << ", " << x.variance() << ")";
  return os;
}

} // namespace histogram
} // namespace boost

#endif
