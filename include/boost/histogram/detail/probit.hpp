// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_PROBIT_HPP
#define BOOST_HISTOGRAM_DETAIL_PROBIT_HPP

#include <cmath>

namespace boost {
namespace histogram {
namespace detail {

template <class T>
T erf_inv_approx(T x){
   T one = T(1);
   T lnx = std::log((one - x)*(one + x));

   T tt1 = T(4.120666747961526) + T(0.5) * lnx;
   T tt2 = T(6.47272819164) * lnx;

   return std::copysign(std::sqrt(-tt1 + std::sqrt(tt1*tt1 - tt2)), x);
}

template <class T>
T probit(T x){
   return T(-1)*(std::sqrt(T(2))*erf_inv_approx((T(2)*((T(1)-x)*T(0.5)))+T(-1))) ;
   // Make it more readable like sqrt_two = std::sqrt(T(2))
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
