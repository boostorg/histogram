// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_fwd_HPP
#define BOOST_fwd_HPP

#include <boost/container/container_fwd.hpp> // for string and new_allocator
#include <boost/histogram/attribute.hpp>     // for BOOST_HISTOGRAM_NODISCARD

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
struct id;
struct log;
struct sqrt;
struct pow;
} // namespace transform

template <typename RealType = double, typename Transform = transform::id,
          typename MetaData = string_type, option_type Options = option_type::uoflow>
class regular;

template <typename RealType = double, typename MetaData = string_type,
          option_type Options = option_type::overflow>
using circular =
    regular<RealType, transform::id, MetaData, Options | option_type::circular>;

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

template <class Axes, class Storage>
class BOOST_HISTOGRAM_NODISCARD grid;

template <class Axes, class Storage = default_storage>
class BOOST_HISTOGRAM_NODISCARD histogram;
} // namespace histogram
} // namespace boost

#endif
