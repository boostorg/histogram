// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_CAT_HPP_
#define _BOOST_HISTOGRAM_DETAIL_CAT_HPP_

#include <boost/config.hpp>
#include <sstream>

namespace boost {
namespace histogram {
namespace detail {
namespace {
BOOST_ATTRIBUTE_UNUSED inline void cat_impl(std::ostringstream&) {}

template <typename T, typename... Ts>
void cat_impl(std::ostringstream& os, const T& t, const Ts&... ts) {
  os << t;
  cat_impl(os, ts...);
}
} // namespace

template <typename... Ts>
std::string cat(const Ts&... args) {
  std::ostringstream os;
  cat_impl(os, args...);
  return os.str();
}
} // namespace detail
} // namespace histogram
} // namespace boost

#endif
