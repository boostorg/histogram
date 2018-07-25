// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_INDEX_CACHE_HPP_
#define _BOOST_HISTOGRAM_DETAIL_INDEX_CACHE_HPP_

#include <cstddef>
#include <memory>

namespace boost {
namespace histogram {
namespace detail {
struct index_cache {
  struct dim_t {
    int idx, size;
    std::size_t stride;
  };

  struct dim_visitor {
    mutable std::size_t stride;
    mutable dim_t* dims;
    template <typename Axis>
    void operator()(const Axis& a) const noexcept {
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

  index_cache(const index_cache& o) : dim_(o.dim_), dims_(new dim_t[o.dim_]) {
    std::copy(o.dims_.get(), o.dims_.get() + dim_, dims_.get());
  }

  index_cache& operator=(const index_cache& o) {
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
}
}
}

#endif
