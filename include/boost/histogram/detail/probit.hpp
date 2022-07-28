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
T erf_inv(T x){
   T tt1, tt2, lnx, sgn;
   sgn = std::copysign(1.0, x);

   x = (T(1) - x)*(T(1) + x);        // x = 1 - x*x;
   lnx = std::log(x);

   tt1 = T(4.120666747961526) + 0.5f * lnx;
   tt2 = T(6.47272819164) * lnx;

   return(sgn*sqrtf(-tt1 + sqrtf(tt1*tt1 - tt2)));
}

template <class T>
T probit(T x){
   return T(-1)*(std::sqrt(T(2))*erf_inv((T(2)*((T(1)-x)*T(0.5)))+T(-1))) ;
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
