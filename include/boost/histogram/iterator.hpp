// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_VALUE_ITERATOR_HPP_
#define _BOOST_HISTOGRAM_VALUE_ITERATOR_HPP_

#include <boost/histogram/detail/index_cache.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mp11.hpp>

namespace boost {
namespace histogram {

template <typename Histogram>
class iterator_over
    : public iterator_facade<iterator_over<Histogram>, typename Histogram::element_type,
                             random_access_traversal_tag,
                             typename Histogram::const_reference> {
public:
  iterator_over(const Histogram& h, std::size_t idx) : histogram_(h), idx_(idx) {}

  iterator_over(const iterator_over& o) : histogram_(o.histogram_), idx_(o.idx_) {}

  iterator_over& operator=(const iterator_over& o) {
    histogram_ = o.histogram_;
    idx_ = o.idx_;
    cache_.reset();
  }

  std::size_t dim() const noexcept { return histogram_.dim(); }

  int idx(std::size_t dim = 0) const noexcept {
    if (!cache_) { cache_.set(histogram_); }
    cache_.set_idx(idx_);
    return cache_.get(dim);
  }

  auto bin() const -> decltype(std::declval<Histogram&>().axis()[0]) {
    return histogram_.axis()[idx()];
  }

  template <std::size_t I>
  auto bin(mp11::mp_size_t<I>) const
      -> decltype(std::declval<Histogram&>().axis(mp11::mp_size_t<I>())[0]) {
    return histogram_.axis(mp11::mp_size_t<I>())[idx(I)];
  }

  template <typename T = Histogram> // use SFINAE for this method
  auto bin(std::size_t dim) const -> decltype(std::declval<T&>().axis(dim)[0]) {
    return histogram_.axis(dim)[idx(dim)];
  }

private:
  bool equal(const iterator_over& rhs) const noexcept {
    return &histogram_ == &rhs.histogram_ && idx_ == rhs.idx_;
  }

  void increment() noexcept { ++idx_; }
  void decrement() noexcept { --idx_; }

  void advance(int n) noexcept { idx_ += n; }

  typename Histogram::const_reference dereference() const noexcept {
    return histogram_.storage_[idx_];
  }

  const Histogram& histogram_;
  std::size_t idx_;
  mutable detail::index_cache cache_;
  friend class ::boost::iterator_core_access;
};

} // namespace histogram
} // namespace boost

#endif
