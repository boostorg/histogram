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

/// Type wrapper to make T as weight
template <typename T>
struct weight_type {
  T value;
};

/// Helper function to mark argument as a weight
template <typename T>
weight_type<T&&> weight(T&& t) {
  return {std::forward<T>(t)};
}

}
}

#endif
