// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_INDEXED_HPP
#define BOOST_HISTOGRAM_INDEXED_HPP

#include <boost/histogram/attribute.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mp11.hpp>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

/// Range over histogram bins with multi-dimensional index.
template <typename Histogram>
class BOOST_HISTOGRAM_NODISCARD indexed_range {
  using axes_type = typename Histogram::axes_type;
  using value_type = typename Histogram::value_type;
  using value_iterator = typename Histogram::const_iterator;
  using axis_data_type = detail::axes_buffer<axes_type, std::pair<std::size_t, int>>;
  using index_type = detail::axes_buffer<axes_type, int>;

public:
  class accessor {
  public:
    int operator[](unsigned d) const { return parent_.index_[d]; }
    auto begin() const { return parent_.index_.begin(); }
    auto end() const { return parent_.index_.end(); }
    auto size() const { return parent_.index_.size(); }

    template <unsigned N>
    decltype(auto) bin(std::integral_constant<unsigned, N>) const {
      return parent_.hist_.axis(std::integral_constant<unsigned, N>())[(*this)[N]];
    }

    decltype(auto) bin(unsigned d) const { return parent_.hist_.axis(d)[(*this)[d]]; }

    double density() const {
      double x = 1;
      auto it = begin();
      parent_.hist_.for_each_axis(
          [&](const auto& a) { x *= axis::traits::width(a, *it++); });
      return *iter_ / x;
    }

    accessor(const indexed_range& parent, value_iterator i) : parent_(parent), iter_(i) {}

    decltype(auto) operator*() const noexcept {
      return *iter_;
    }

    decltype(auto) operator->() const noexcept {
      return iter_;
    }

  private:
    const indexed_range& parent_;
    value_iterator iter_;
  };

  class const_iterator
      : public boost::iterator_facade<const_iterator, accessor,
                                      boost::forward_traversal_tag, accessor> {
  public:
    const_iterator(const indexed_range& p, value_iterator i) noexcept
        : parent_(p), iter_(i) {}

  protected:
    void increment() noexcept {
      parent_.include_extra_bins_ ? iter_over_all() : iter_over_inner();
    }

    void iter_over_all() noexcept {
      auto k = ++iter_ - parent_.hist_.begin();
      auto s = parent_.axis_data_.end();
      auto i = parent_.index_.end();
      while (s != parent_.axis_data_.begin()) {
        --s;
        --i;
        *i = k / s->first;
        k %= s->first;
        if (*i == s->second) *i = -1;
      }
    }

    void iter_over_inner() noexcept {
      auto s = parent_.axis_data_.begin();
      auto i = parent_.index_.begin();
      ++*i;
      while (*i == s->second) {
        if (i == (parent_.index_.end() - 1)) {
          *i = s->first;
          break;
        } else {
          *i = 0;
          ++*(++i);
          ++s;
        }
      }

      std::size_t k = 0, stride = 1;
      s = parent_.axis_data_.begin();
      for (auto x : parent_.index_) {
        k += x * stride;
        stride *= s->first;
        ++s;
      }
      iter_ = parent_.hist_.begin() + k;
    }

    bool equal(const_iterator rhs) const noexcept {
      return &parent_ == &rhs.parent_ && iter_ == rhs.iter_;
    }

    accessor dereference() const noexcept { return {parent_, iter_}; }

    friend class ::boost::iterator_core_access;

  private:
    const indexed_range& parent_;
    value_iterator iter_;
    bool skip_;
  };

  indexed_range(const Histogram& h, bool include_extra_bins)
      : hist_(h)
      , include_extra_bins_(include_extra_bins)
      , axis_data_(h.rank())
      , index_(h.rank(), 0) {
    auto it = axis_data_.begin();
    std::size_t s = 1;
    h.for_each_axis([&](const auto& a) {
      if (include_extra_bins_) {
        it->first = s;
        s *= axis::traits::extend(a);
        it->second = axis::traits::underflow_index(a);
      } else {
        it->first = axis::traits::extend(a);
        it->second = a.size();
      }
      ++it;
    });
  }

  const_iterator begin() const { return {*this, hist_.begin()}; }
  const_iterator end() const { return {*this, hist_.end()}; }

private:
  const Histogram& hist_;
  const bool include_extra_bins_;
  axis_data_type axis_data_;
  mutable index_type index_;
}; // namespace histogram

template <typename Histogram>
indexed_range<detail::unqual<Histogram>> indexed(Histogram&& h,
                                                 bool include_extra_bins = false) {
  return {std::forward<Histogram>(h), include_extra_bins};
}

} // namespace histogram
} // namespace boost

#endif
