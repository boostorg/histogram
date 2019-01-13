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
#include <boost/histogram/fwd.hpp>
#include <boost/iterator/iterator_adaptor.hpp>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

/// Range over histogram bins with multi-dimensional index.
template <class Histogram>
class BOOST_HISTOGRAM_NODISCARD indexed_range {
  using histogram_iterator = std::conditional_t<std::is_const<Histogram>::value,
                                                typename Histogram::const_iterator,
                                                typename Histogram::iterator>;
  struct cache_item {
    int idx, begin, end, extend;
  };
  using cache_type =
      detail::stack_buffer<cache_item, typename detail::naked<Histogram>::axes_type>;

public:
  class accessor {
  public:
    class index_iterator
        : public boost::iterator_adaptor<index_iterator,
                                         typename cache_type::const_iterator> {
    public:
      index_iterator(typename cache_type::const_iterator i)
          : index_iterator::iterator_adaptor_(i) {}
      decltype(auto) operator*() const noexcept { return index_iterator::base()->idx; }
    };

    int operator[](unsigned d) const { return parent_->cache_[d].idx; }
    auto begin() const { return index_iterator(parent_->cache_.begin()); }
    auto end() const { return index_iterator(parent_->cache_.end()); }
    auto size() const { return parent_->cache_.size(); }

    template <unsigned N>
    decltype(auto) bin(std::integral_constant<unsigned, N>) const {
      return parent_->hist_.axis(std::integral_constant<unsigned, N>())[(*this)[N]];
    }

    decltype(auto) bin(unsigned d) const { return parent_->hist_.axis(d)[(*this)[d]]; }

    double density() const {
      double x = 1;
      auto it = begin();
      parent_->hist_.for_each_axis([&](const auto& a) {
        const auto w = axis::traits::width_as<double>(a, *it++);
        x *= w ? w : 1;
      });
      return *iter_ / x;
    }

    accessor(indexed_range* parent, histogram_iterator i) : parent_(parent), iter_(i) {}

    decltype(auto) operator*() const noexcept { return *iter_; }
    auto operator-> () const noexcept { return iter_; }

  private:
    indexed_range* parent_;
    histogram_iterator iter_;
  };

  class range_iterator
      : public boost::iterator_adaptor<range_iterator, histogram_iterator, accessor,
                                       boost::forward_traversal_tag, accessor> {
  public:
    range_iterator(indexed_range* p, histogram_iterator i) noexcept
        : range_iterator::iterator_adaptor_(i), parent_(p) {}

    accessor operator*() const noexcept { return {parent_, range_iterator::base()}; }

  private:
    friend class boost::iterator_core_access;

    void increment() noexcept {
      std::size_t stride = 1;
      auto c = parent_->cache_.begin();
      ++c->idx;
      ++range_iterator::base_reference();
      while (c->idx == c->end && ((c + 1) != parent_->cache_.end())) {
        c->idx = c->begin;
        range_iterator::base_reference() -= (c->end - c->begin) * stride;
        stride *= c->extend;
        ++c;
        ++c->idx;
        range_iterator::base_reference() += stride;
      }
    }

    mutable indexed_range* parent_;
  };

  indexed_range(Histogram& h, bool include_extra_bins)
      : hist_(h)
      , include_extra_bins_(include_extra_bins)
      , begin_(hist_.begin())
      , end_(begin_)
      , cache_(hist_.rank()) {
    auto c = cache_.begin();
    std::size_t stride = 1;
    h.for_each_axis([&, this](const auto& a) {
      const auto opt = axis::traits::options(a);
      const auto shift = test(opt, axis::option::underflow);

      c->extend = axis::traits::extend(a);
      c->begin = include_extra_bins_ ? -shift : 0;
      c->end = c->extend - shift -
               (include_extra_bins_ ? 0 : test(opt, axis::option::overflow));
      c->idx = c->begin;

      begin_ += (c->begin + shift) * stride;
      if ((c + 1) < cache_.end())
        end_ += (c->begin + shift) * stride;
      else
        end_ += (c->end + shift) * stride;

      stride *= c->extend;
      ++c;
    });
  }

  range_iterator begin() noexcept { return {this, begin_}; }
  range_iterator end() noexcept { return {this, end_}; }

private:
  Histogram& hist_;
  const bool include_extra_bins_;
  histogram_iterator begin_, end_;
  mutable cache_type cache_;
}; // namespace histogram

template <typename Histogram>
indexed_range<std::remove_reference_t<Histogram>> indexed(
    Histogram&& h, bool include_extra_bins = false) {
  return {std::forward<Histogram>(h), include_extra_bins};
}

} // namespace histogram
} // namespace boost

#endif
