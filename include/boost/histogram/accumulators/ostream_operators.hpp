// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_OSTREAM_OPERATORS_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_OSTREAM_OPERATORS_HPP

#include <boost/histogram/histogram_fwd.hpp>
#include <ostream>

namespace boost {
namespace histogram {
namespace accumulators {
template <typename CharT, typename Traits, typename W>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const neumaier<W>& x) {
  os << "neumaier(" << x.large() << " + " << x.small() << ")";
  return os;
}

template <typename CharT, typename Traits, typename W>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const weight<W>& x) {
  os << "weight(" << x.value() << ", " << x.variance() << ")";
  return os;
}

template <typename CharT, typename Traits, typename W>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const mean<W>& x) {
  os << "mean(" << x.sum() << ", " << x.value() << ", " << x.variance() << ")";
  return os;
}

template <typename CharT, typename Traits, typename W>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const weighted_mean<W>& x) {
  os << "weighted_mean(" << x.sum() << ", " << x.sum2() << ", " << x.value() << ", "
     << x.variance() << ")";
  return os;
}
} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif
