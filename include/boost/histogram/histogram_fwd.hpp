// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP
#define BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP

#include <boost/config.hpp>
#include <boost/container/container_fwd.hpp> // for string and new_allocator

// Why boost containers as defaults and not std containers?
// - std::vector does not work with incomplete types, but boost::container::vector does
// - std::string is very large on MSVC, boost::container::string is small everywhere

namespace boost {
namespace histogram {

namespace axis {

/// empty metadata type
struct null_type {};

/// default metadata type
using string_type = boost::container::string;

enum class option_type;

constexpr inline option_type operator|(option_type a, option_type b) {
  return static_cast<option_type>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr inline bool operator&(option_type a, option_type b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

enum class option_type {
  none = 0,
  underflow = 1 << 0,
  overflow = 1 << 1,
  uoflow = underflow | overflow,
  circular = 1 << 3,
};

namespace transform {
template <typename T = double>
struct identity;
template <typename T = double>
struct log;
template <typename T = double>
struct sqrt;
template <typename T = double>
struct pow;
} // namespace transform

template <typename TransformOrRealType = double, typename MetaData = string_type,
          option_type Options = option_type::uoflow>
class regular;

template <typename RealType = double, typename MetaData = string_type,
          option_type Options = option_type::overflow>
using circular = regular<RealType, MetaData, Options | option_type::circular>;

template <typename IntType = int, typename MetaData = string_type,
          option_type Options = option_type::underflow | option_type::overflow>
class integer;

template <typename RealType = double, typename MetaData = string_type,
          option_type Options = option_type::uoflow,
          typename Allocator = boost::container::new_allocator<RealType>>
class variable;

template <typename T = int, typename MetaData = string_type,
          option_type Options = option_type::overflow,
          typename Allocator = boost::container::new_allocator<T>>
class category;

template <typename... Ts>
class variant;
} // namespace axis

template <typename T>
struct weight_type;

template <typename T>
struct sample_type;

namespace accumulators {
template <typename RealType = double>
class sum;
template <typename RealType = double>
class weighted_sum;
template <typename RealType = double>
class mean;
template <typename RealType = double>
class weighted_mean;
} // namespace accumulators

struct unsafe_access;

template <typename T>
struct storage_adaptor;

template <typename Allocator = boost::container::new_allocator<void>>
struct adaptive_storage;

using default_storage = adaptive_storage<>;
using weight_storage =
    storage_adaptor<boost::container::vector<accumulators::weighted_sum<>>>;
using profile_storage = storage_adaptor<boost::container::vector<accumulators::mean<>>>;
using weighted_profile_storage =
    storage_adaptor<boost::container::vector<accumulators::weighted_mean<>>>;

template <class Axes, class Storage = default_storage>
class histogram;
} // namespace histogram
} // namespace boost

#endif
