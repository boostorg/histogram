// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_WEIGHT_HPP
#define BOOST_HISTOGRAM_WEIGHT_HPP

#include <utility>

namespace boost {
namespace histogram {

/// Type envelope to mark value as weight
template <typename T>
struct weight_type {
  T value;
};

/// Helper function to mark argument as weight
template <typename T>
auto weight(T&& t) {
  return weight_type<T>{std::forward<T>(t)};
}

} // namespace histogram
} // namespace boost

#endif
