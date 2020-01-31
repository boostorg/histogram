// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_VARIANT2_WITH_WARNINGS_DISABLED_HPP
#define BOOST_HISTOGRAM_DETAIL_VARIANT2_WITH_WARNINGS_DISABLED_HPP

#include <boost/config/workaround.hpp>
#if BOOST_WORKAROUND(BOOST_GCC, >= 5)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#endif
#include <boost/variant2/variant.hpp>
#if BOOST_WORKAROUND(BOOST_GCC, >= 5)
#pragma GCC diagnostic pop
#endif

#endif
