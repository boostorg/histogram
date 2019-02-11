// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_FWD_HPP
#define BOOST_HISTOGRAM_FWD_HPP

/**
  \file boost/histogram/fwd.hpp
  Forward declarations, basic typedefs, and default template arguments for main classes.
*/

#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/detail/attribute.hpp> // BOOST_HISTOGRAM_DETAIL_NODISCARD
#include <string>
#include <vector>

namespace boost {
namespace histogram {
namespace axis {

/// Integral type for axis indices
using index_type = int;

/// Real type for axis indices
using real_index_type = double;

/// empty metadata type
struct null_type {};

/// default metadata type
using default_metadata = std::string;

namespace transform {
struct id;
struct log;
struct sqrt;
struct pow;
} // namespace transform

template <class Value = double, class Transform = transform::id,
          class MetaData = default_metadata, option Options = option::use_default>
class regular;

template <class Value = double, class MetaData = default_metadata,
          option Options = option::overflow>
using circular = regular<Value, transform::id, MetaData, join(Options, option::circular)>;

template <class Value = int, class MetaData = default_metadata,
          option Options = option::use_default>
class integer;

template <class Value = double, class MetaData = default_metadata,
          option Options = option::use_default, class Allocator = std::allocator<Value>>
class variable;

template <class Value = int, class MetaData = default_metadata,
          option Options = option::overflow, class Allocator = std::allocator<Value>>
class category;

template <class... Ts>
class variant;
} // namespace axis

template <class T>
struct weight_type;

template <class T>
struct sample_type;

namespace accumulators {
template <class Value = double>
class sum;
template <class Value = double>
class weighted_sum;
template <class Value = double>
class mean;
template <class Value = double>
class weighted_mean;
} // namespace accumulators

struct unsafe_access;

template <class Allocator = std::allocator<char>>
class unlimited_storage;

template <class T>
class storage_adaptor;

template <class T, class A = std::allocator<T>>
using dense_storage = storage_adaptor<std::vector<T, A>>;

using default_storage = unlimited_storage<>;

using weight_storage = dense_storage<accumulators::weighted_sum<>>;

using profile_storage = dense_storage<accumulators::mean<>>;

using weighted_profile_storage = dense_storage<accumulators::weighted_mean<>>;

template <class Axes, class Storage = default_storage>
class BOOST_HISTOGRAM_DETAIL_NODISCARD histogram;
} // namespace histogram
} // namespace boost

#endif
