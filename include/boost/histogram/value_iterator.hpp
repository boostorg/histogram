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

template <typename Storage>
class value_iterator_over
    : public iterator_facade<
          value_iterator_over<Storage>, typename Storage::value_type,
          forward_traversal_tag, typename Storage::value_type> {
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
  /// begin iterator
  template <typename Histogram>
  value_iterator_over(const Histogram &h, const Storage &s) : s_(s), idx_(0) {
    dims_.reserve(h.dim());
    h.for_each_axis(dim_visitor{1, dims_});
  }

  /// end iterator
  explicit value_iterator_over(const Storage &s)
      : s_(s), idx_(std::numeric_limits<std::size_t>::max()) {}

  value_iterator_over(const value_iterator_over &) = default;
  value_iterator_over &operator=(const value_iterator_over &) = default;

  int idx(int dim = 0) const noexcept { return dims_[dim].idx; }

private:
  void increment() noexcept {
    auto iter = dims_.begin();
    while (iter->idx == (iter->size - 1)) {
      iter->idx = 0;
      ++iter;
    }
    if (iter == dims_.end())
      idx_ = std::numeric_limits<std::size_t>::max();
    else {
      ++(iter->idx);
      idx_ = 0;
      for (; iter != dims_.end(); ++iter)
        idx_ += iter->idx * iter->stride;
    }
  }
  bool equal(const value_iterator_over &other) const noexcept {
    return &s_ == &(other.s_) && idx_ == other.idx_;
  }
  typename Storage::value_type dereference() const { return s_.value(idx_); }

  const Storage &s_;
  std::size_t idx_;
  std::vector<dim_t> dims_;
  friend class ::boost::iterator_core_access;
};
} // namespace histogram
} // namespace boost

#endif