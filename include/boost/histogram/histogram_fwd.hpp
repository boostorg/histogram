// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_

#include <boost/histogram/storage/adaptive_storage.hpp>
#include <set>
#include <type_traits>

namespace boost {
namespace histogram {

using Static = std::integral_constant<int, 0>;
using Dynamic = std::integral_constant<int, 1>;

template <class Variant, class Axes, class Storage = adaptive_storage<>>
class histogram;

class weight {
public:
  explicit weight(double v) : value(v) {}
  explicit operator double() const { return value; }

private:
  double value;
};

class count {
public:
  explicit count(unsigned v) : value(v) {}
  explicit operator unsigned() const { return value; }

private:
  unsigned value;
};

namespace detail {
template <unsigned> struct keep_remove : std::set<unsigned> {
  keep_remove(const std::initializer_list<unsigned> &l)
      : std::set<unsigned>(l) {}
};
} // namespace detail

using keep = detail::keep_remove<0>;
using remove = detail::keep_remove<1>;

} // namespace histogram
} // namespace boost

#endif
