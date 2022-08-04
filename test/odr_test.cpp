// Copyright 2019 Henry Schreiner, Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// The header windows.h and possibly others illegially do the following
#define small char
// which violates the C++ standard. We make sure here that including our headers work
// nevertheless by avoiding the preprocessing token `small`. For more details, see
// https://github.com/boostorg/histogram/issues/342

// include all Boost.Histogram header here; see odr_main_test.cpp for details
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/serialization.hpp>

#include <boost/histogram/detail/ignore_deprecation_warning_begin.hpp>
#include <boost/histogram/detail/ignore_deprecation_warning_end.hpp>
#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <boost/histogram/utility/agresti_coull_interval.hpp>
#include <boost/histogram/utility/clopper_pearson_interval.hpp>
#include <boost/histogram/utility/jeffreys_interval.hpp>
#include <boost/histogram/utility/wald_interval.hpp>
#include <boost/histogram/utility/wilson_interval.hpp>
#include <boost/histogram/detail/normal.hpp>
#include <boost/histogram/detail/erf_inv.hpp>
