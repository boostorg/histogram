// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_VALUE_ITERATOR_HPP_
#define _BOOST_HISTOGRAM_VALUE_ITERATOR_HPP_

#include <array>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mp11.hpp>
#include <limits>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {

template <typename Histogram>
class iterator_over
    : public iterator_facade<
          iterator_over<Histogram>, typename Histogram::element_type,
          random_access_traversal_tag, typename Histogram::const_reference> {
public:
  iterator_over(const Histogram& h, std::size_t idx)
      : histogram_(h), idx_(idx) {}

  iterator_over(const iterator_over&) = default;
  iterator_over& operator=(const iterator_over&) = default;

  unsigned dim() const noexcept { return histogram_.dim(); }

  int idx(unsigned dim = 0) const noexcept {
    histogram_.index_cache_(idx_);
    return histogram_.index_cache_[dim];
  }

  auto bin() const
      -> decltype(std::declval<Histogram&>().axis(mp11::mp_int<0>())[0]) {
    return histogram_.axis(mp11::mp_int<0>())[idx(0)];
  }

  template <int Dim>
  auto bin(mp11::mp_int<Dim>) const
      -> decltype(std::declval<Histogram&>().axis(mp11::mp_int<Dim>())[0]) {
    return histogram_.axis(mp11::mp_int<Dim>())[idx(Dim)];
  }

  template <typename T = Histogram> // use SFINAE for this method
  auto bin(unsigned dim) const -> decltype(std::declval<T&>().axis(dim)[0]) {
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
  friend class ::boost::iterator_core_access;
};

} // namespace histogram
} // namespace boost

#endif
