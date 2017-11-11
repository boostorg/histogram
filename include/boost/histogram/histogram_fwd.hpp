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

} // namespace histogram
} // namespace boost

#endif
