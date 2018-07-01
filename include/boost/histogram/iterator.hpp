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

struct dim_visitor {
  mutable std::size_t stride;
  mutable dim_t *dims;
  template <typename Axis> void operator()(const Axis &a) const noexcept {
    *dims++ = dim_t{0, a.size(), stride};
    stride *= a.shape();
  }
};

void decode(std::size_t idx, unsigned dim, dim_t* dims) {
  dims += dim;
  while ((--dims, --dim)) {
    dims->idx = idx / dims->stride;
    idx -= dims->idx * dims->stride;
    dims->idx -= (dims->idx > dims->size) * (dims->size + 2);
  }
  dims->idx = idx;
  dims->idx -= (dims->idx > dims->size) * (dims->size + 2);
}

class multi_index {
public:
  unsigned dim() const noexcept { return dim_; }

  int idx(unsigned dim=0) const noexcept {
    if (idx_ != last_) {
      const_cast<std::size_t&>(last_) = idx_;
      decode(idx_, dim_, const_cast<dim_t*>(dims_.get()));
    }
    return dims_[dim].idx;
  }

protected:
  multi_index() = default;

  template <typename Histogram>
  multi_index(const Histogram &h, std::size_t idx) :
    dim_(h.dim()), idx_(idx), last_(idx),
    dims_(new dim_t[h.dim()]) {
    h.for_each_axis(dim_visitor{1, dims_.get()});
    if (0 < idx && idx < h.bincount()) {
      decode(idx, dim_, dims_.get());
    }
  }

  multi_index(const multi_index& o) :
    dim_(o.dim_), idx_(o.idx_), last_(o.last_),
    dims_(new dim_t[o.dim_])
  {
    std::copy(o.dims_.get(), o.dims_.get() + o.dim_, dims_.get());
  }

  multi_index& operator=(const multi_index& o) {
    if (this != &o) {
      if (dim_ != o.dim_) {
        dims_.reset(new dim_t[o.dim_]);
      }
      dim_ = o.dim_;
      idx_ = o.idx_;
      last_ = o.last_;
      std::copy(o.dims_.get(), o.dims_.get() + o.dim_, dims_.get());
    }
    return *this;
  }

  multi_index(multi_index&&) = default;
  multi_index& operator=(multi_index&&) = default;

  void increment() noexcept { ++idx_; }
  void decrement() noexcept { --idx_; }

  unsigned dim_;
  std::size_t idx_;
  std::size_t last_;
  std::unique_ptr<dim_t[]> dims_;
};
} // namespace detail

template <typename Histogram, typename Storage>
class iterator_over
    : public iterator_facade<
          iterator_over<Histogram, Storage>, typename Storage::element_type,
          bidirectional_traversal_tag, typename Storage::const_reference>,
      public detail::multi_index {

public:
  iterator_over(const Histogram &h, const Storage &s, std::size_t idx)
      : detail::multi_index(h, idx), histogram_(h), storage_(s) {}

  iterator_over(const iterator_over &) = default;
  iterator_over &operator=(const iterator_over &) = default;
  iterator_over(iterator_over &&) = default;
  iterator_over &operator=(iterator_over &&) = default;

  auto bin() const
      -> decltype(std::declval<Histogram &>().axis(mpl::int_<0>())[0]) {
    return histogram_.axis(mpl::int_<0>())[dims_[0].idx];
  }

  template <int N>
  auto bin(mpl::int_<N>) const
      -> decltype(std::declval<Histogram &>().axis(mpl::int_<N>())[0]) {
    return histogram_.axis(mpl::int_<N>())[dims_[N].idx];
  }

  template <typename Int> // TODO: solve this with a std::enable_if
  auto bin(Int dim) const
      -> decltype(std::declval<Histogram &>().axis(dim)[0]) {
    return histogram_.axis(dim)[dims_[dim].idx];
  }

private:
  bool equal(const iterator_over &rhs) const noexcept {
    return &storage_ == &rhs.storage_ && idx_ == rhs.idx_;
  }
  typename Storage::const_reference dereference() const noexcept {
    return storage_[idx_];
  }

  const Histogram &histogram_;
  const Storage &storage_;
  friend class ::boost::iterator_core_access;
};

} // namespace histogram
} // namespace boost

#endif
