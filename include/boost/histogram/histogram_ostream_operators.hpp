// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_OSTREAM_OPERATORS_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_OSTREAM_OPERATORS_HPP_

#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/histogram/histogram_fwd.hpp>
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

template <typename D, typename A, typename S>
inline std::ostream &operator<<(std::ostream &os, const histogram<D, A, S> &h) {
  os << "histogram(";
  detail::axis_ostream_visitor sh(os);
  h.for_each_axis(sh);
  os << (h.dim() ? "\n)" : ")");
  return os;
}

} // namespace histogram
} // namespace boost

#endif
