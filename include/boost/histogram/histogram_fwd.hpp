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

/* Most of the histogram code is generic and works for any number of axes. Buffers with a
 * fixed maximum capacity are used in some places, which have a size equal to the rank of
 * a histogram. The buffers are statically allocated to improve performance, which means
 * that they need a preset maximum capacity. 32 seems like a safe upper limit for the rank
 * (you can nevertheless increase it here if necessary): the simplest non-trivial axis has
 * 2 bins; even if counters are used which need only a byte of storage per bin, this still
 * corresponds to 4 GB of storage.
 */
BOOST_ATTRIBUTE_UNUSED static constexpr unsigned limit =
#ifdef BOOST_HISTOGRAM_AXES_LIMIT
    BOOST_HISTOGRAM_AXES_LIMIT;
#else
    32;
#endif

/// empty metadata type
struct null_type {};

enum class option_type {
  none = 0,
  underflow = 1,
  overflow = 2,
  uoflow = 3,
};

constexpr inline option_type operator|(option_type a, option_type b) {
  return static_cast<option_type>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr inline bool operator&(option_type a, option_type b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

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

template <typename Transform = transform::identity<double>,
          typename MetaData = boost::container::string,
          option_type Options = option_type::uoflow>
class regular;

template <typename RealType = double, typename MetaData = boost::container::string,
          option_type Options = option_type::overflow>
class circular;

template <typename RealType = double,
          typename Allocator = boost::container::new_allocator<RealType>,
          typename MetaData = boost::container::string,
          option_type Options = option_type::uoflow>
class variable;

template <typename IntType = double, typename MetaData = boost::container::string,
          option_type Options = option_type::uoflow>
class integer;

template <typename T = int, typename Allocator = boost::container::new_allocator<T>,
          typename MetaData = boost::container::string,
          option_type Options = option_type::overflow>
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

template <class Axes, class Storage = default_storage>
class histogram;
} // namespace histogram
} // namespace boost

#endif
