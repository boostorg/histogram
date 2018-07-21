// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_INDEX_MAPPER_HPP_
#define _BOOST_HISTOGRAM_DETAIL_INDEX_MAPPER_HPP_

#include <algorithm>
#include <cstddef>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {
struct index_mapper {
  std::size_t first = 0, second = 0;

  index_mapper(const std::vector<unsigned>& nvec,
               const std::vector<bool>& bvec) {
    dims.reserve(nvec.size());
    std::size_t s1 = 1, s2 = 1;
    auto bi = bvec.begin();
    for (const auto& ni : nvec) {
      if (*bi) {
        dims.push_back({s1, s2});
        s2 *= ni;
      } else {
        dims.push_back({s1, 0});
      }
      s1 *= ni;
      ++bi;
    }
    std::sort(dims.begin(), dims.end(), [](const dim& a, const dim& b) {
      return a.stride1 > b.stride1;
    });
    nfirst = s1;
  }

  bool next() {
    ++first;
    second = 0;
    auto f = first;
    for (const auto& d : dims) {
      auto i = f / d.stride1;
      f -= i * d.stride1;
      second += i * d.stride2;
    }
    return first < nfirst;
  }

private:
  std::size_t nfirst;
  struct dim {
    std::size_t stride1, stride2;
  };
  std::vector<dim> dims;
};
}
}
}

#endif
