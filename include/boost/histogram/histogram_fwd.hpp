// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_

#include <boost/histogram/detail/meta.hpp>
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
template <typename Ns> struct keep_static { using type = Ns; };
struct keep_dynamic : std::set<unsigned> {};
} // namespace detail

template <typename... Ns>
auto keep(Ns...)
    -> detail::keep_static<typename detail::unique_sorted<Ns...>::type> {
  return {};
}

template <typename Iterator, typename = detail::is_iterator<Iterator>>
detail::keep_dynamic keep(Iterator begin, Iterator end) {
  detail::keep_dynamic s;
  std::copy(begin, end, s.begin());
  return s;
}

} // namespace histogram
} // namespace boost

#endif
