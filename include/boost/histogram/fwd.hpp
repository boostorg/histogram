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
using default_metadata = boost::container::string;

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

template <class RealType = double, class Transform = transform::id,
          class MetaData = default_metadata, option_type Options = option_type::uoflow>
class regular;

template <class RealType = double, class MetaData = default_metadata,
          option_type Options = option_type::overflow>
using circular =
    regular<RealType, transform::id, MetaData, Options | option_type::circular>;

template <class IntType = int, class MetaData = default_metadata,
          option_type Options = option_type::underflow | option_type::overflow>
class integer;

template <class RealType = double, class MetaData = default_metadata,
          option_type Options = option_type::uoflow,
          class Allocator = boost::container::new_allocator<RealType>>
class variable;

template <class T = int, class MetaData = default_metadata,
          option_type Options = option_type::overflow,
          class Allocator = boost::container::new_allocator<T>>
class category;

template <class... Ts>
class variant;
} // namespace axis

template <class T>
struct weight_type;

template <class T>
struct sample_type;

namespace accumulators {
template <class RealType = double>
class sum;
template <class RealType = double>
class weighted_sum;
template <class RealType = double>
class mean;
template <class RealType = double>
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
