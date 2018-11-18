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

struct index_mapper_item {
  std::size_t stride[2];
};

class index_mapper
    : public boost::container::static_vector<index_mapper_item, axis::limit> {
public:
  std::size_t total = 1;

  index_mapper(unsigned dim) : static_vector(dim) {}

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
};

struct index_mapper_reduce_item {
  std::size_t stride[2];
  int underflow[2], overflow[2], begin, end, merge;
};

class index_mapper_reduce
    : public boost::container::static_vector<index_mapper_reduce_item, axis::limit> {
public:
  std::size_t total = 1;

  index_mapper_reduce(unsigned dim) : static_vector(dim) {}

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
};
} // namespace detail
} // namespace histogram
} // namespace boost

#endif
