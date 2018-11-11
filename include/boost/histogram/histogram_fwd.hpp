// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP
#define BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP

#include <memory> // for std::allocator
#include <string>

namespace boost {
namespace histogram {
namespace axis {
struct empty_metadata_type {};

enum class option_type {
  none = 0,
  overflow = 1,
  underflow_and_overflow = 2,
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

template <typename Transform = transform::identity<double>,
          typename MetaData = std::string>
class regular;

template <typename RealType = double, typename MetaData = std::string>
class circular;

template <typename RealType = double, typename Allocator = std::allocator<RealType>,
          typename MetaData = std::string>
class variable;

template <typename IntType = double, typename MetaData = std::string>
class integer;

template <typename T = int, typename Allocator = std::allocator<T>,
          typename MetaData = std::string>
class category;

template <typename... Ts>
class variant;
} // namespace axis

template <typename T>
struct weight_type;

template <typename T>
struct sample_type;

namespace accumulators {
template <typename RealType>
class weight;
template <typename RealType>
class mean;
} // namespace accumulators

struct unsafe_access;

template <typename T>
struct storage_adaptor;

template <typename Allocator = std::allocator<char>>
struct adaptive_storage;

using default_storage = adaptive_storage<>;

template <class Axes, class Storage = default_storage>
class histogram;
} // namespace histogram
} // namespace boost

#endif
