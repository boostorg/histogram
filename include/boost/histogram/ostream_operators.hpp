// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_OSTREAM_OPERATORS_HPP_
#define _BOOST_HISTOGRAM_OSTREAM_OPERATORS_HPP_

#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <ostream>

namespace boost {
namespace histogram {

namespace detail {
struct axis_ostream_visitor {
  std::ostream &os_;
  explicit axis_ostream_visitor(std::ostream &os) : os_(os) {}
  template <typename Axis> void operator()(const Axis &a) const {
    os_ << "\n  " << a << ",";
  }
};
} // namespace detail

template <typename... Ts>
inline std::ostream &operator<<(std::ostream &os, const histogram<Ts...> &h) {
  os << "histogram(";
  h.for_each_axis(detail::axis_ostream_visitor(os));
  os << (h.dim() ? "\n)" : ")");
  return os;
}

template <typename T>
inline std::ostream &operator<<(std::ostream& os, const weight_counter<T>& x) {
  os << "weight_counter(" << x.value() << ", " << x.variance() << ")";
  return os;
}

} // namespace histogram
} // namespace boost

#endif
