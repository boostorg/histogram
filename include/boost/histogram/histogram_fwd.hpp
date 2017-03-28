// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_

#include <boost/histogram/storage/adaptive_storage.hpp>

namespace boost {
namespace histogram { 

template <bool Dynamic,
          class Axes,
          class Storage = adaptive_storage<> > class histogram;

} // namespace histogram
} // namespace boost

#endif
