// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_WEIGHT_HPP_
#define _BOOST_HISTOGRAM_DETAIL_WEIGHT_HPP_

namespace boost {
namespace histogram {
namespace detail {

/// Used by nstore to hold a sum of weighted counts and a variance estimate
struct weight_t {
  double w, w2;
  weight_t() = default;
  weight_t(const weight_t&) = default;
  weight_t(weight_t&&) = default;
  weight_t& operator=(const weight_t&) = default;
  weight_t& operator=(weight_t&&) = default;
  weight_t& operator+=(const weight_t& o)
  { w += o.w; w2 += o.w2; return *this; }
  weight_t& operator++()
  { ++w; ++w2; return *this; }
  bool operator==(const weight_t& o) const
  { return w == o.w && w2 == o.w2; }
  bool operator!=(const weight_t& o) const
  { return !operator==(o); }
  weight_t& add_weight(double v)
  { w += v; w2 += v*v; return *this; }

  explicit weight_t(int v) : w(v), w2(v) {}

  template <typename T>
  weight_t& operator=(T v)
  { w = static_cast<double>(v); w2 = static_cast<double>(v);  return *this; }
  template <typename T>
  weight_t& operator+=(T v)
  { w += static_cast<double>(v); w2 += static_cast<double>(v); return *this; }
  template <typename T>
  bool operator==(T v) const
  { return w == static_cast<double>(v) && w2 == static_cast<double>(v); }
  template <typename T>
  bool operator!=(T v) const
  { return !operator==(v); }
};

}
}
}

#endif
