// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_HPP_
#define BOOST_HISTOGRAM_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/histogram/utility.hpp>
#include <boost/histogram/static_histogram.hpp>
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/container_storage.hpp>

/**
 * \file boost/histogram.hpp
   \brief Includes all the non-experimental headers of the Boost.histogram library.

   The library consists of two histogram implementations
   \ref boost::histogram::static_histogram "static_histogram"
   and \ref boost::histogram::dynamic_histogram "dynamic_histogram"
   which share a common interface. The first is faster, but lacks run-time
   polymorphism, the second implements the opposite trade-off.
   Several axis types are included, which implement different binning algorithms.
   The axis types are passed in the constructor of the histogram to configure
   its binning.
*/

#endif
