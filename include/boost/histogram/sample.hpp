// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SAMPLE_HPP
#define BOOST_HISTOGRAM_SAMPLE_HPP

#include <tuple>
#include <utility>

namespace boost {
namespace histogram {

/// Type envelope to make value as sample
template <typename T>
struct sample_type {
  T value;
};

/// Helper function to mark arguments as sample
template <typename... Ts>
sample_type<std::tuple<Ts&&...>> sample(Ts&&... ts) noexcept {
  return {std::forward_as_tuple(std::forward<Ts>(ts)...)};
}

} // namespace histogram
} // namespace boost

#endif
