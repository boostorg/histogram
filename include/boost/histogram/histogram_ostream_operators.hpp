// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_STREAMER_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_STREAMER_HPP_

#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/histogram/static_histogram.hpp>
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/utility.hpp>
#include <ostream>

namespace boost {
namespace histogram {

namespace detail {
  struct axis_ostream_visitor : public static_visitor<void> {
    std::ostream& os_;
    axis_ostream_visitor(std::ostream& os) : os_(os) {}
    template <typename Axis>
    void operator()(const Axis& a) const
    { os_ << "\n  " << a << ",";  }
  };
}

template <template <typename, typename> typename Histogram, typename Axes, typename Storage,
          typename = detail::is_histogram<Histogram<Axes, Storage>>>
inline std::ostream& operator<<(std::ostream& os, const Histogram<Axes, Storage>& h)
{
    os << "histogram(";
    detail::axis_ostream_visitor sh(os);
    for_each_axis(h, sh);
    os << (h.dim() ? "\n)" : ")");
    return os;
}

} // NS histogram
} // NS boost

#endif
