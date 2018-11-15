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

namespace boost {
namespace histogram {
namespace detail {
class index_mapper
    : public boost::container::static_vector<std::pair<std::size_t, std::size_t>,
                                             axis::limit> {
public:
  std::size_t first = 0, second = 0, ntotal = 1;

  using static_vector<std::pair<std::size_t, std::size_t>, axis::limit>::static_vector;

  bool next() {
    ++first;
    second = 0;
    auto f = first;
    for (auto it = end(); it != begin(); --it) {
      const auto& d = *(it - 1);
      // compiler usually optimizes div & mod into one div
      second += f / d.first * d.second;
      f %= d.first;
    }
    return first < ntotal;
  }
};
} // namespace detail
} // namespace histogram
} // namespace boost

#endif
