// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_VALUE_ITERATOR_HPP_
#define _BOOST_HISTOGRAM_VALUE_ITERATOR_HPP_

#include <boost/iterator/iterator_facade.hpp>
#include <limits>
#include <vector>

namespace boost {
namespace histogram {

namespace detail {
class multi_index {
  struct dim_t {
    int idx, size;
    std::size_t stride;
  };

  struct dim_visitor {
    mutable std::size_t stride;
    std::vector<dim_t> &dims;
    template <typename Axis> void operator()(const Axis &a) const {
      dims.emplace_back(dim_t{0, a.size(), stride});
      stride *= a.shape();
    }
  };

public:
  multi_index() = default;

  template <typename Histogram>
  explicit multi_index(const Histogram &h) : idx_(0) {
    dims_.reserve(h.dim());
    h.for_each_axis(dim_visitor{1, dims_});
  }

  int idx(unsigned dim = 0) const noexcept { return dims_[dim].idx; }
  unsigned dim() const noexcept { return dims_.size(); }

protected:
  void increment() noexcept {
    auto iter = dims_.begin();
    for (; iter != dims_.end(); ++iter) {
      ++(iter->idx);
      if (iter->idx < iter->size)
        break;
      iter->idx = 0;
    }
    if (iter == dims_.end())
      idx_ = std::numeric_limits<std::size_t>::max();
    else {
      idx_ = 0;
      for (; iter != dims_.end(); ++iter)
        idx_ += iter->idx * iter->stride;
    }
  }

  std::size_t idx_ = std::numeric_limits<std::size_t>::max();
  std::vector<dim_t> dims_;
};
} // namespace detail

template <typename Storage>
class bin_iterator_over
    : public iterator_facade<
          bin_iterator_over<Storage>, typename Storage::bin_type,
          forward_traversal_tag, typename Storage::bin_type>,
      public detail::multi_index {

public:
  /// begin iterator
  template <typename Histogram>
  bin_iterator_over(const Histogram &h, const Storage &s)
      : detail::multi_index(h), s_(s) {}

  /// end iterator
  explicit bin_iterator_over(const Storage &s) : s_(s) {}

  bin_iterator_over(const bin_iterator_over &) = default;
  bin_iterator_over &operator=(const bin_iterator_over &) = default;

private:
  bool equal(const bin_iterator_over &other) const noexcept {
    return &s_ == &(other.s_) && idx_ == other.idx_;
  }
  typename Storage::bin_type dereference() const { return s_[idx_]; }

  const Storage &s_;
  friend class ::boost::iterator_core_access;
};
} // namespace histogram
} // namespace boost

#endif
