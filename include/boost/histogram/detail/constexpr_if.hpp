// Copyright 2018-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_CONSTEXPR_IF_HPP
#define BOOST_HISTOGRAM_DETAIL_CONSTEXPR_IF_HPP

#include <boost/config.hpp>

#ifdef BOOST_NO_CXX17_IF_CONSTEXPR
#define BOOST_HISTOGRAM_CONSTEXPR_IF if
#else
#define BOOST_HISTOGRAM_CONSTEXPR_IF if constexpr
#endif

#endif
