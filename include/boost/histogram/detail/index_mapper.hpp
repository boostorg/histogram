// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_INDEX_MAPPER_HPP
#define BOOST_HISTOGRAM_DETAIL_INDEX_MAPPER_HPP

#include <boost/container/static_vector.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <cstddef>
#include <functional>

namespace boost {
namespace histogram {
namespace detail {

template <typename A>
struct index_mapper {
  struct item {
    std::size_t stride[2];
  };
  using buffer_type = axes_buffer<A, item>;

  index_mapper(unsigned dim) : buffer(dim) {}

  template <typename T, typename U>
  void operator()(T& dst, const U& src) {
    for (std::size_t i = 0; i < total; ++i) {
      std::size_t j = 0;
      auto imod = i;
      for (auto it = end(); it != begin(); --it) {
        const auto& d = *(it - 1);
        // compiler usually optimizes div & mod into one div
        const auto k = imod / d.stride[0];
        imod %= d.stride[0];
        j += k * d.stride[1];
      }
      dst.add(j, src[i]);
    }
  }

  decltype(auto) begin() { return buffer.begin(); }
  decltype(auto) end() { return buffer.end(); }

  decltype(auto) operator[](unsigned i) { return buffer[i]; }

  buffer_type buffer;
  std::size_t total = 1;
};

template <typename A>
class index_mapper_reduce {
public:
  struct item {
    std::size_t stride[2];
    int underflow[2], overflow[2], begin, end, merge;
  };
  using buffer_type = axes_buffer<A, item>;

  index_mapper_reduce(unsigned dim) : buffer(dim) {}

  template <typename T, typename U>
  void operator()(T& dst, const U& src) {
    for (std::size_t i = 0; i < total; ++i) {
      std::size_t j = 0;
      auto imod = i;
      bool drop = false;
      for (auto it = end(); it != begin(); --it) {
        const auto& d = *(it - 1);
        // compiler usually optimizes div & mod into one div
        int k = imod / d.stride[0];
        imod %= d.stride[0];
        if (k < d.begin || k == d.underflow[0]) {
          k = d.underflow[1];
        } else if (k >= d.end || k == d.overflow[0]) {
          k = d.overflow[1];
        } else {
          k = (k - d.begin) / d.merge;
        }
        drop |= k < 0;
        j += k * d.stride[1];
      }
      if (!drop) dst.add(j, src[i]);
    }
  }

  decltype(auto) begin() { return buffer.begin(); }
  decltype(auto) end() { return buffer.end(); }

  decltype(auto) operator[](unsigned i) { return buffer[i]; }

  buffer_type buffer;
  std::size_t total = 1;
};
} // namespace detail
} // namespace histogram
} // namespace boost

#endif
