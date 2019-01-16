// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_fwd_HPP
#define BOOST_fwd_HPP

#include <boost/container/container_fwd.hpp> // for string and new_allocator
#include <boost/histogram/attribute.hpp>     // for BOOST_HISTOGRAM_NODISCARD

// Why boost containers and not std containers?
// - boost has the static_vector specialization, which is internally used a lot
// - std::string is very large on MSVC, boost::container::string is small everywhere

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
using default_metadata = boost::container::string;

enum class option {
  none = 0,
  underflow = 0b1,
  overflow = 0b10,
  circular = 0b100,
  growth = 0b1000,
  use_default = static_cast<int>(underflow) | static_cast<int>(overflow),
};

constexpr inline option operator~(option a) {
  return static_cast<option>(~static_cast<int>(a));
}

constexpr inline option operator&(option a, option b) {
  return static_cast<option>(static_cast<int>(a) & static_cast<int>(b));
}

constexpr inline option operator|(option a, option b) {
  return static_cast<option>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr bool test(option a, option b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

constexpr inline option join(option a, option b) {
  // growth turns off *flow and circular, circular turns off underflow, underflow turns
  // off circular and growth, overflow turns of growth
  a = a | b;
  if (test(b, option::underflow)) a = a & ~option::circular & ~option::growth;
  if (test(b, option::overflow)) a = a & ~option::growth;
  if (test(b, option::circular)) a = a & ~option::underflow & ~option::growth;
  if (test(b, option::growth)) return option::growth;
  return a;
}

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
          option Options = option::use_default,
          class Allocator = boost::container::new_allocator<Value>>
class variable;

template <class Value = int, class MetaData = default_metadata,
          option Options = option::overflow,
          class Allocator = boost::container::new_allocator<Value>>
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

template <class Allocator = boost::container::new_allocator<void>>
struct adaptive_storage;

template <class T>
struct storage_adaptor;

template <class T, class A = boost::container::new_allocator<T>>
using dense_storage = storage_adaptor<boost::container::vector<T, A>>;

using default_storage = adaptive_storage<>;

using weight_storage = dense_storage<accumulators::weighted_sum<>>;

using profile_storage = dense_storage<accumulators::mean<>>;

using weighted_profile_storage = dense_storage<accumulators::weighted_mean<>>;

template <class Axes, class Storage = default_storage>
class BOOST_HISTOGRAM_NODISCARD histogram;
} // namespace histogram
} // namespace boost

#endif
