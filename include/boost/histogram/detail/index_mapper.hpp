// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_INDEX_MAPPER_HPP
#define BOOST_HISTOGRAM_DETAIL_INDEX_MAPPER_HPP

#include <array>
#include <cstddef>

namespace boost {
namespace histogram {
namespace detail {
class index_mapper : public std::array<std::pair<std::size_t, std::size_t>, 32> {
public:
  std::size_t first = 0, second = 0, ntotal = 1;

  index_mapper(std::size_t n) : dims_end(begin() + n) {}

  bool next() {
    ++first;
    second = 0;
    auto f = first;
    for (auto it = end(); it != begin(); --it) {
      const auto& d = *(it - 1);
      auto i = f / d.first;
      f -= i * d.first;
      second += i * d.second;
    }
    return first < ntotal;
  }

  iterator end() { return dims_end; }

private:
  iterator dims_end;
};
} // namespace detail
} // namespace histogram
} // namespace boost

#endif
