// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_
#define _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_

#include <algorithm>
#include <boost/histogram/detail/meta.hpp>
#include <boost/utility/string_view.hpp>
#include <ostream>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

inline void escape(std::ostream &os, const string_view s) {
  os << '\'';
  for (auto sit = s.begin(); sit != s.end(); ++sit) {
    if (*sit == '\'' && (sit == s.begin() || *(sit - 1) != '\\')) {
      os << "\\\'";
    } else {
      os << *sit;
    }
  }
  os << '\'';
}

template <typename Axis>
inline void lin(std::size_t &out, std::size_t &stride, const Axis &a,
                int j) noexcept {
  // the following is highly optimized code that runs in a hot loop;
  // please measure the performance impact of changes
  const int uoflow = a.uoflow();
  // set stride to zero if 'j' is not in range,
  // this communicates the out-of-range condition to the caller
  stride *= (j >= -uoflow) & (j < (a.size() + uoflow));
  j += (j < 0) * (a.size() + 2); // wrap around if in < 0
  out += j * stride;
#pragma GCC diagnostic ignored "-Wstrict-overflow"
  stride *= a.shape();
}

template <typename Axis>
inline void xlin(std::size_t &out, std::size_t &stride, const Axis &a,
                 const typename Axis::value_type &x) noexcept {
  // the following is highly optimized code that runs in a hot loop;
  // please measure the performance impact of changes
  int j = a.index(x);
  // j is guaranteed to be in range [-1, bins]
  j += (j < 0) * (a.size() + 2); // wrap around if j < 0
  out += j * stride;
#pragma GCC diagnostic ignored "-Wstrict-overflow"
  stride *= (j < a.shape()) * a.shape(); // stride == 0 indicates out-of-range
}

struct index_mapper {
  std::size_t first = 0, second = 0;

  index_mapper(const std::vector<unsigned> &nvec,
               const std::vector<bool> &bvec) {
    dims.reserve(nvec.size());
    std::size_t s1 = 1, s2 = 1;
    auto bi = bvec.begin();
    for (const auto &ni : nvec) {
      if (*bi) {
        dims.push_back({s1, s2});
        s2 *= ni;
      } else {
        dims.push_back({s1, 0});
      }
      s1 *= ni;
      ++bi;
    }
    std::sort(dims.begin(), dims.end(), [](const dim &a, const dim &b) {
      if (a.stride1 == b.stride1)
        return 0;
      return a.stride1 < b.stride1 ? -1 : 1;
    });
    nfirst = s1;
  }

  bool next() {
    ++first;
    second = 0;
    auto f = first;
    for (const auto &d : dims) {
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

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
