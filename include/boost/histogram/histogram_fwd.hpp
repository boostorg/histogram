// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_

#include <boost/histogram/detail/meta.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <set>

namespace boost {
namespace histogram {

using Static = std::integral_constant<int, 0>;
using Dynamic = std::integral_constant<int, 1>;

class adaptive_storage;

namespace axis {

namespace transform {
  struct identity;
  struct log;
  struct sqrt;
  struct cos;
  struct pow;
}

template <typename RealType = double, typename Transform = transform::identity> class regular;
template <typename RealType = double> class circular;
template <typename RealType = double> class variable;
template <typename IntType = int> class integer;
template <typename T = int> class category;

using builtins =
    mpl::vector<axis::regular<>, axis::regular<double, axis::transform::log>,
                axis::regular<double, axis::transform::sqrt>,
                axis::regular<double, axis::transform::cos>,
                axis::regular<double, axis::transform::pow>, axis::circular<>,
                axis::variable<>, axis::integer<>,
                axis::category<>, axis::category<std::string>>;
}

template <class Variant, class Axes, class Storage = adaptive_storage>
class histogram;

struct weight {
  weight(double w) : value(w) {}
  double value;
};

struct count {
  count(unsigned n) : value(n) {}
  unsigned value;
};

namespace detail {
template <typename T> struct keep_static {};
template <typename T> struct remove_static {};

struct keep_dynamic : public std::set<unsigned> {
  using base_type = std::set<unsigned>;
  using base_type::base_type;
};
struct remove_dynamic : public std::set<unsigned> {
  using base_type = std::set<unsigned>;
  using base_type::base_type;
};

inline void insert(keep_dynamic &) {} // end recursion
template <typename... Rest>
inline void insert(keep_dynamic &s, unsigned i, Rest... rest) {
  s.insert(i);
  insert(s, rest...);
}
} // namespace detail

// for static and dynamic histogram
template <int N, typename... Rest>
inline auto keep(mpl::int_<N>, Rest...)
    -> detail::keep_static<detail::unique_sorted<mpl::vector<mpl::int_<N>, Rest...>>> {
  return {};
}

template <typename Iterator, typename = detail::is_iterator<Iterator>>
inline detail::keep_dynamic keep(Iterator begin, Iterator end) {
  return {begin, end};
}

template <typename... Rest>
inline detail::keep_dynamic keep(unsigned i, Rest... rest) {
  detail::keep_dynamic s;
  detail::insert(s, i, rest...);
  return s;
}

} // namespace histogram
} // namespace boost

#endif
