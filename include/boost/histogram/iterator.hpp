// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_VALUE_ITERATOR_HPP_
#define _BOOST_HISTOGRAM_VALUE_ITERATOR_HPP_

#include <array>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mpl/int.hpp>
#include <limits>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {

namespace detail {

struct dim_t {
  int idx, size;
  std::size_t stride;
};

std::size_t inc(dim_t *iter, dim_t *end) noexcept {
  for (; iter != end; ++iter) {
    ++(iter->idx);
    if (iter->idx < iter->size)
      break;
    iter->idx = 0;
  }

  if (iter == end)
    return std::numeric_limits<std::size_t>::max();

  std::size_t idx = 0;
  for (; iter != end; ++iter)
    idx += iter->idx * iter->stride;
  return idx;
}

struct dim_visitor {
  mutable std::size_t stride;
  mutable dim_t *dims;
  template <typename Axis> void operator()(const Axis &a) const noexcept {
    *dims++ = dim_t{0, a.size(), stride};
    stride *= a.shape();
  }
};

class multi_index {
public:
  int idx(unsigned dim = 0) const noexcept { return dims_[dim].idx; }
  unsigned dim() const noexcept { return dims_.size(); }

protected:
  multi_index() = default;

  template <typename Histogram>
  explicit multi_index(const Histogram &h) : idx_(0), dims_(h.dim()) {
    h.for_each_axis(dim_visitor{1, dims_.data()});
  }

  void increment() noexcept {
    idx_ = inc(dims_.data(), dims_.data() + dims_.size());
  }

  std::size_t idx_;
  std::vector<dim_t> dims_;
};
} // namespace detail

template <typename Histogram, typename Storage>
class iterator_over
    : public iterator_facade<
          iterator_over<Histogram, Storage>, typename Storage::element_type,
          forward_traversal_tag, typename Storage::const_reference>,
      public detail::multi_index {

public:
  /// begin iterator
  iterator_over(const Histogram &h, const Storage &s, bool)
      : detail::multi_index(h), histogram_(h), storage_(s) {}

  /// end iterator
  iterator_over(const Histogram &h, const Storage &s)
      : histogram_(h), storage_(s) {
    idx_ = std::numeric_limits<std::size_t>::max();
  }

  iterator_over(const iterator_over &) = default;
  iterator_over &operator=(const iterator_over &) = default;

  auto bin() const
      -> decltype(std::declval<Histogram &>().axis(mpl::int_<0>())[0]) {
    return histogram_.axis(mpl::int_<0>())[dims_[0].idx];
  }

  template <int N>
  auto bin(mpl::int_<N>) const
      -> decltype(std::declval<Histogram &>().axis(mpl::int_<N>())[0]) {
    return histogram_.axis(mpl::int_<N>())[dims_[N].idx];
  }

  template <typename Int>
  auto bin(Int dim) const
      -> decltype(std::declval<Histogram &>().axis(dim)[0]) {
    return histogram_.axis(dim)[dims_[dim].idx];
  }

private:
  bool equal(const iterator_over &rhs) const noexcept {
    return &storage_ == &rhs.storage_ && idx_ == rhs.idx_;
  }
  typename Storage::const_reference dereference() const {
    return storage_[idx_];
  }

  const Histogram &histogram_;
  const Storage &storage_;
  friend class ::boost::iterator_core_access;
};

} // namespace histogram
} // namespace boost

#endif
