// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SAMPLE_HPP
#define BOOST_HISTOGRAM_SAMPLE_HPP

#include <utility>

namespace boost {
namespace histogram {

/// Type envelope to make value as sample
template <typename T>
struct sample_type {
  T value;
};

/// Helper function to mark argument as sample
template <typename T>
sample_type<T> sample(T&& t) {
  return {std::forward<T>(t)};
}

} // namespace histogram
} // namespace boost

#endif
