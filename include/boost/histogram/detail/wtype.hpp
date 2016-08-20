// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_WTYPE_HPP_
#define _BOOST_HISTOGRAM_DETAIL_WTYPE_HPP_

#include <boost/cstdint.hpp>
#include <type_traits>
#include <ostream>

namespace boost {
namespace histogram {
namespace detail {

/// Used by nstore to hold a sum of weighted counts and a variance estimate
struct wtype {
  double w, w2;
  wtype() = default;  
  wtype(const wtype&) = default;
  wtype(wtype&&) = default;
  wtype& operator=(const wtype&) = default;
  wtype& operator=(wtype&&) = default;
  wtype(double v) : w(v), w2(v) {}
  wtype& add_weight(double v)
  { w += v; w2 += v*v; return *this; }
  wtype& operator+=(double v)
  { w += v; w2 += v; return *this; }
  wtype& operator+=(const wtype& o)
  { w += o.w; w2 += o.w2; return *this; }
  wtype& operator++()
  { ++w; ++w2; return *this; }
  bool operator==(double v) const
  { return w == v && w2 == v; }
  bool operator!=(double v) const
  { return w != v || w2 != v; }
  bool operator==(const wtype& o) const
  { return w == o.w && w2 == o.w2; }
  bool operator!=(const wtype& o) const
  { return w != o.w || w2 != o.w2; }
};

inline
std::ostream& operator<<(std::ostream& os, const wtype& w)
{
  os << '(' << w.w << ',' << w.w2 << ')';
  return os;
}

}
}
}

#endif
