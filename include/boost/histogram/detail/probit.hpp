// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_PROBIT_HPP
#define BOOST_HISTOGRAM_DETAIL_PROBIT_HPP

#include <cmath>

# define PI 3.14159265358979323846  /* pi */

namespace boost {
namespace histogram {
namespace detail {

template <class T>
T ErfInv(T x){
   T tt1, tt2, lnx, sgn;
   sgn = (x < 0) ? -1.0f : 1.0f;

   x = (1 - x)*(1 + x);        // x = 1 - x*x;
   lnx = logf(x);

   tt1 = 2/(PI*0.15449436008930206298828125) + 0.5f * lnx;
   tt2 = 1/(0.15449436008930206298828125) * lnx;

   return(sgn*sqrtf(-tt1 + sqrtf(tt1*tt1 - tt2)));
}

template <class T>
T probit(T x){
   return (-1)*((std::pow(2,0.5))*ErfInv((2*((1-x)/2))-1)) ;
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
