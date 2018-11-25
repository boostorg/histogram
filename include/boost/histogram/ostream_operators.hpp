// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_OSTREAM_OPERATORS_HPP
#define BOOST_HISTOGRAM_OSTREAM_OPERATORS_HPP

#include <boost/histogram/histogram_fwd.hpp>
#include <ostream>

namespace boost {
namespace histogram {

template <typename CharT, typename Traits, typename A, typename S>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const histogram<A, S>& h) {
  os << "histogram(";
  h.for_each_axis([&](const auto& a) { os << "\n  " << a << ","; });
  os << (h.rank() ? "\n)" : ")");
  return os;
}

} // namespace histogram
} // namespace boost

#endif
