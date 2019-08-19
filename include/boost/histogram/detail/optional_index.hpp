// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_OPTIONAL_INDEX_HPP
#define BOOST_HISTOGRAM_DETAIL_OPTIONAL_INDEX_HPP

#include <boost/assert.hpp>

namespace boost {
namespace histogram {
namespace detail {

// integer with a persistent invalid state, similar to NaN
struct optional_index {
  static constexpr auto invalid = ~static_cast<std::size_t>(0);

  optional_index& operator=(const std::size_t x) noexcept {
    value = x;
    return *this;
  }

  optional_index& operator+=(const std::size_t x) noexcept {
    BOOST_ASSERT(x != invalid);
    if (value != invalid) { value += x; }
    return *this;
  }

  optional_index& operator+=(const optional_index& x) noexcept {
    if (x.valid()) return operator+=(x.value);
    value = invalid;
    return *this;
  }

  bool valid() const noexcept { return value != invalid; }
  const std::size_t& operator*() const noexcept { return value; }

  std::size_t value;
};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
