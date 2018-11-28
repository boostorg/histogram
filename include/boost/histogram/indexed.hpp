// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_INDEXED_HPP
#define BOOST_HISTOGRAM_INDEXED_HPP

#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/nodiscard.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mp11.hpp>
#include <utility>

namespace boost {
namespace histogram {

/// Range over histogram bins with multi-dimensional index.
template <typename Histogram>
class indexed_range {
  using storage_type = typename Histogram::storage_type;
  using axes_type = typename Histogram::axes_type;

  struct stride_t {
    std::size_t stride;
    int underflow;
  };
  using strides_type = detail::axes_buffer<axes_type, stride_t>;
  using index_type = detail::axes_buffer<axes_type, int>;
  using value_type = decltype(std::declval<storage_type&>()[0]);

public:
  class index_value : public index_type {
  public:
    template <std::size_t N>
    decltype(auto) bin(mp11::mp_size_t<N>) const {
      return detail::axis_get<N>(axes_)[(*this)[N]];
    }

    decltype(auto) bin(unsigned d) const {
      return detail::axis_get(axes_, d)[(*this)[d]];
    }

    double density() const {
      double x = 1;
      auto it = this->begin();
      detail::for_each_axis(axes_,
                            [&](const auto& a) { x *= axis::traits::width(a, *it++); });
      return value / x;
    }

    const value_type value;

    index_value(const axes_type& a, value_type v)
        : index_type(detail::axes_size(a)), value(v), axes_(a) {}

  private:
    const axes_type& axes_;
  };

  class const_iterator
      : public boost::iterator_facade<const_iterator, index_value,
                                      boost::random_access_traversal_tag, index_value> {
  public:
    const_iterator(const indexed_range& parent, std::size_t idx) noexcept
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
    index_value dereference() const noexcept {
      auto result = index_value(parent_.axes_, parent_.storage_[idx_]);
      auto sit = parent_.strides_.end();
      auto iit = result.end();
      auto idx = idx_;
      while (sit != parent_.strides_.begin()) {
        --sit;
        --iit;
        *iit = idx / sit->stride;
        idx %= sit->stride;
        if (*iit == sit->underflow) *iit = -1;
      }
      return result;
    }

    friend class ::boost::iterator_core_access;

  private:
    const indexed_range& parent_;
    std::size_t idx_;
  };

  indexed_range(const Histogram& h)
      : axes_(unsafe_access::axes(h))
      , storage_(unsafe_access::storage(h))
      , strides_(h.rank()) {
    auto it = strides_.begin();
    std::size_t s = 1;
    h.for_each_axis([&](const auto& a) {
      it->stride = s;
      s *= axis::traits::extend(a);
      it->underflow = axis::traits::underflow_index(a);
      ++it;
    });
  }

  const_iterator begin() const { return {*this, 0}; }
  const_iterator end() const { return {*this, storage_.size()}; }

private:
  const axes_type& axes_;
  const storage_type& storage_;
  strides_type strides_;
};

BOOST_HISTOGRAM_DETAIL_NODISCARD
template <typename Histogram>
indexed_range<detail::unqual<Histogram>> indexed(Histogram&& h) {
  return {std::forward<Histogram>(h)};
}

} // namespace histogram
} // namespace boost

#endif
