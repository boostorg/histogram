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

struct state_t {
  unsigned dim;
  std::size_t idx;
};

struct dim_t {
  int idx, size;
  std::size_t stride;
};

union block_t {
  state_t state;
  dim_t dim;
};

struct index_cache : public std::unique_ptr<block_t[]> {
  using ptr_t = std::unique_ptr<block_t[]>;

  struct dim_visitor {
    mutable std::size_t stride;
    mutable block_t* b;
    template <typename Axis>
    void operator()(const Axis& a) const noexcept {
      b->dim = dim_t{0, a.size(), stride};
      ++b;
      stride *= a.shape();
    }
  };

  template <typename H>
  void set(const H& h) {
    if (!(*this) || h.dim() != ptr_t::get()->state.dim) {
      ptr_t::reset(new block_t[h.dim() + 1]);
      ptr_t::get()->state.dim = h.dim();
      ptr_t::get()->state.idx = 0;
    }
    h.for_each_axis(dim_visitor{1, ptr_t::get() + 1});
  }

  void set_idx(std::size_t idx) {
    auto& s = ptr_t::get()->state;
    if (idx == s.idx) return;
    s.idx = idx;
    auto d = s.dim;
    auto b = (ptr_t::get() + 1) + d;
    while ((--b, --d)) {
      b->dim.idx = idx / b->dim.stride;
      idx -= b->dim.idx * b->dim.stride;
      b->dim.idx -= (b->dim.idx > b->dim.size) * (b->dim.size + 2);
    }
    b->dim.idx = idx;
    b->dim.idx -= (b->dim.idx > b->dim.size) * (b->dim.size + 2);
  }

  int get(unsigned d) const { return (ptr_t::get() + 1 + d)->dim.idx; }
};
}
}
}

#endif
