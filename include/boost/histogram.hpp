// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_HPP_
#define BOOST_HISTOGRAM_HPP_

#include <boost/histogram/axis/axis.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>

/**
 * \file boost/histogram.hpp
   \brief Includes all standard headers of the Boost.histogram library.

   The library consists of two histogram implementations, a static and a
   dynamic one. If you need to configure histograms at runtime, choose the
   dynamic one, otherwise use the static one. The static one is faster,
   and a little more convenient, since some errors can be caught at compile
   time and you don't need to worry about type conversions.

   You can freely mix the implementations in your code, they are convertible
   into each other, and there is a common interface for both. To the extend
   of what is technically possible, the library has been designed to allow
   you to write code that is agnostic of the implementation variant.
*/

#endif
