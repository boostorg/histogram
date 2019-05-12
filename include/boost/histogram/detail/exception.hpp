// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/config.hpp>

#ifdef BOOST_NO_EXCEPTIONS

#define BOOST_HISTOGRAM_DETAIL_TRY
#define BOOST_HISTOGRAM_DETAIL_CATCH_ANY if (false)
#define BOOST_HISTOGRAM_DETAIL_RETHROW

#else

#define BOOST_HISTOGRAM_DETAIL_TRY try
#define BOOST_HISTOGRAM_DETAIL_CATCH_ANY catch (...)
#define BOOST_HISTOGRAM_DETAIL_RETHROW throw

#endif
