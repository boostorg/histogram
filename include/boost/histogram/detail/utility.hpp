// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_
#define _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/type_index.hpp>
#include <boost/utility/string_view.hpp>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <memory>

namespace boost {
namespace histogram {
namespace detail {

// two_pi can be found in boost/math, but it is defined here to reduce deps
constexpr double two_pi = 6.283185307179586;

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

// the following is highly optimized code that runs in a hot loop;
// please measure the performance impact of changes
inline void lin(std::size_t &out, std::size_t &stride, const int axis_size,
                const int axis_shape, int j) noexcept {
  BOOST_ASSERT_MSG(stride == 0 || (-1 <= j && j <= axis_size),
                   "index must be in bounds for this algorithm");
  j += (j < 0) * (axis_size + 2); // wrap around if j < 0
  out += j * stride;
#ifndef _MSC_VER
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif
  stride *= (j < axis_shape) * axis_shape; // stride == 0 indicates out-of-range
}

struct index_cache {
  struct dim_t {
    int idx, size;
    std::size_t stride;
  };

  struct dim_visitor {
    mutable std::size_t stride;
    mutable dim_t *dims;
    template <typename Axis> void operator()(const Axis &a) const noexcept {
      *dims++ = dim_t{0, a.size(), stride};
      stride *= a.shape();
    }
  };

  unsigned dim_ = 0;
  std::size_t idx_ = 0;
  std::unique_ptr<dim_t[]> dims_;

  index_cache() = default;
  index_cache(index_cache&&) = default;
  index_cache& operator=(index_cache&&) = default;

  index_cache(const index_cache& o) :
    dim_(o.dim_), dims_(new dim_t[o.dim_])
  {
    std::copy(o.dims_.get(), o.dims_.get() + dim_, dims_.get());
  }

  index_cache& operator=(const index_cache& o)
  {
    if (this != &o) {
      if (o.dim_ != dim_) {
        dim_ = o.dim_;
        dims_.reset(new dim_t[dim_]);
      }
      std::copy(o.dims_.get(), o.dims_.get() + dim_, dims_.get());
    }
    return *this;
  }

  template <typename H>
  void reset(const H& h) {
    if (h.dim() != dim_) {
      dim_ = h.dim();
      dims_.reset(new dim_t[dim_]);
    }
    h.for_each_axis(dim_visitor{1, dims_.get()});
  }

  void operator()(std::size_t idx) {
    if (idx == idx_) return;
    idx_ = idx;
    auto dim_ptr = dims_.get();
    auto dim = dim_;
    dim_ptr += dim;
    while ((--dim_ptr, --dim)) {
      dim_ptr->idx = idx / dim_ptr->stride;
      idx -= dim_ptr->idx * dim_ptr->stride;
      dim_ptr->idx -= (dim_ptr->idx > dim_ptr->size) * (dim_ptr->size + 2);
    }
    dim_ptr->idx = idx;
    dim_ptr->idx -= (dim_ptr->idx > dim_ptr->size) * (dim_ptr->size + 2);
  }

  int operator[](unsigned dim) const { return dims_[dim].idx; }
};

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
    std::sort(dims.begin(), dims.end(),
              [](const dim &a, const dim &b) { return a.stride1 > b.stride1; });
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

template <typename T>
inline typename std::enable_if<(is_castable_to_int<T>::value), int>::type
indirect_int_cast(T &&t) noexcept {
  return static_cast<int>(std::forward<T>(t));
}

template <typename T>
inline typename std::enable_if<!(is_castable_to_int<T>::value), int>::type
indirect_int_cast(T &&) noexcept {
  // Cannot use static_assert here, because this function is created as a
  // side-effect of TMP. It must be valid at compile-time.
  BOOST_ASSERT_MSG(false, "bin argument not convertible to int");
  return 0;
}

template <typename S, typename T>
inline void fill_storage(S &s, std::size_t idx, weight<T> &&w) {
  s.add(idx, w);
}

template <typename S> inline void fill_storage(S &s, std::size_t idx) {
  s.increase(idx);
}

template <typename S>
inline auto storage_get(const S &s, std::size_t idx, bool error) ->
    typename S::const_reference {
  if (error)
    throw std::out_of_range("bin index out of range");
  return s[idx];
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
