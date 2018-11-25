// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_INDEXED_HPP
#define BOOST_HISTOGRAM_INDEXED_HPP

#include <boost/container/static_vector.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <utility>

namespace boost {
namespace histogram {

/// Range over histogram bins with multi-dimensional index.
template <typename Histogram>
class indexed_type {
  using storage_type = typename Histogram::storage_type;
  using index_type = boost::container::static_vector<int, axis::limit>;
  struct stride_t {
    std::size_t stride;
    int underflow;
  };
  using strides_type = boost::container::static_vector<stride_t, axis::limit>;

public:
  using value_type =
      std::pair<const index_type&, decltype(std::declval<storage_type&>()[0])>;

  class const_iterator
      : public boost::iterator_facade<const_iterator, value_type,
                                      boost::random_access_traversal_tag, value_type> {
  public:
    const_iterator(const indexed_type& parent, std::size_t idx) noexcept
        : parent_(parent), idx_(idx) {}

  protected:
    void increment() noexcept { ++idx_; }
    void decrement() noexcept { --idx_; }
    void advance(std::ptrdiff_t n) noexcept { idx_ += n; }
    std::ptrdiff_t distance_to(const_iterator rhs) const noexcept {
      return rhs.idx_ - idx_;
    }
    bool equal(const_iterator rhs) const noexcept {
      return &parent_ == &rhs.parent_ && idx_ == rhs.idx_;
    }
    value_type dereference() const noexcept {
      auto sit = parent_.strides_.end();
      auto iit = parent_.index_.end();
      auto idx = idx_;
      while (sit != parent_.strides_.begin()) {
        --sit;
        --iit;
        *iit = idx / sit->stride;
        idx %= sit->stride;
        if (*iit == sit->underflow) *iit = -1;
      }
      return {parent_.index_, parent_.storage_[idx_]};
    }

    friend class ::boost::iterator_core_access;

  private:
    const indexed_type& parent_;
    std::size_t idx_;
  };

  indexed_type(const Histogram& h)
      : storage_(unsafe_access::storage(h)), strides_(h.rank()), index_(h.rank()) {
    auto it = strides_.begin();
    std::size_t s = 1;
    h.for_each_axis([&](const auto& a) {
      it->stride = s;
      s *= axis::traits::extend(a);
      it->underflow = axis::traits::underflow_index(a);
      ++it;
    });
  }

  indexed_type(const indexed_type&) = default;
  indexed_type& operator=(const indexed_type& rhs) = default;

  const_iterator begin() const { return {*this, 0}; }
  const_iterator end() const { return {*this, storage_.size()}; }

private:
  const storage_type& storage_;
  strides_type strides_;
  mutable index_type index_;
};

template <typename Histogram>
indexed_type<detail::unqual<Histogram>> indexed(Histogram&& h) {
  return {std::forward<Histogram>(h)};
}

} // namespace histogram
} // namespace boost

#endif
