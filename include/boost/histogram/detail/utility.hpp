// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_
#define _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_

#include <ostream>

namespace boost {
namespace histogram {
namespace detail {

template <typename String>
inline void escape(std::ostream &os, const String &s) {
  os << '\'';
  for (auto sit = s.begin(); sit != s.end(); ++sit) {
    if (*sit == '\'' && (sit == s.begin() || *(sit - 1) != '\\')) {
      os << "\\\'";
    } else {
      os << *sit;
    }
  }
  os << '\'';
}

template <typename A, typename> struct lin {
  void operator()(std::size_t &out, std::size_t &stride, const A &a, int j) {
    // the following is highly optimized code that runs in a hot loop;
    // please measure the performance impact of changes
    const int uoflow = a.uoflow();
    // set stride to zero if 'j' is not in range,
    // this communicates the out-of-range condition to the caller
    stride *= (j >= -uoflow) & (j < (a.bins() + uoflow));
    j += (j < 0) * (a.bins() + 2); // wrap around if in < 0
    out += j * stride;
#pragma GCC diagnostic ignored "-Wstrict-overflow"
    stride *= a.shape();
  }
};

template <typename A, typename T> struct xlin {
  void operator()(std::size_t &out, std::size_t &stride, const A &a,
                  const T &x) {
    // the following is highly optimized code that runs in a hot loop;
    // please measure the performance impact of changes
    int j = a.index(x);
    // j is guaranteed to be in range [-1, bins]
    j += (j < 0) * (a.bins() + 2); // wrap around if j < 0
    out += j * stride;
#pragma GCC diagnostic ignored "-Wstrict-overflow"
    stride *= (j < a.shape()) * a.shape(); // stride == 0 indicates out-of-range
  }
};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
