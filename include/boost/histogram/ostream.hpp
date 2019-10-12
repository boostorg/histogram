// Copyright 2015-2019 Hans Dembinski
// Copyright (c) 2019 Przemyslaw Bartosik
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_OSTREAM_HPP
#define BOOST_HISTOGRAM_OSTREAM_HPP

#include <algorithm> // max_element
#include <boost/histogram.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/fwd.hpp>
#include <cmath>   // floor, pow
#include <iomanip> // setw
#include <iosfwd>
#include <iostream> // cout
#include <limits>   // infinity

/**
  \file boost/histogram/ostream.hpp

  A simple streaming operator for the histogram type. The text representation is
  rudimentary and not guaranteed to be stable between versions of Boost.Histogram. This
  header is not included by any other header and must be explicitly included to use the
  streaming operator.

  To you use your own, simply include your own implementation instead of this header.
 */

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED

namespace boost {
namespace histogram {

template <typename CharT, typename Traits, typename A, typename S>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const histogram<A, S>& h) {
  os << "histogram(";
  h.for_each_axis([&](const auto& a) { os << "\n  " << a << ","; });
  std::size_t i = 0;
  for (auto&& x : h) os << "\n  " << i++ << ": " << x;
  os << (h.rank() ? "\n)" : ")");
  return os;
}

} // namespace histogram
} // namespace boost

#endif // BOOST_HISTOGRAM_DOXYGEN_INVOKED

#endif
