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

enum class coverage { inner, all, use_default = inner };

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
    class indices_view {
    public:
      class index_iterator
          : public boost::iterator_adaptor<index_iterator,
                                           typename cache_type::const_iterator> {
      public:
        index_iterator(typename cache_type::const_iterator i)
            : index_iterator::iterator_adaptor_(i) {}
        decltype(auto) operator*() const noexcept { return index_iterator::base()->idx; }
      };

      auto begin() const { return index_iterator(cache_.begin()); }
      auto end() const { return index_iterator(cache_.end()); }
      auto size() const { return cache_.size(); }
      int operator[](unsigned d) const { return cache_[d].idx; }
      int at(unsigned d) const { return cache_.at(d).idx; }

    private:
      indices_view(const cache_type& c) : cache_(c) {}
      const cache_type& cache_;
      friend class accessor;
    };

    // pointer interface for value
    decltype(auto) operator*() const noexcept { return *iter_; }
    decltype(auto) get() const noexcept { return *iter_; }
    decltype(auto) operator-> () const noexcept { return iter_; }

    // convenience interface
    int index(unsigned d = 0) const { return parent_->cache_[d].idx; }
    auto indices() const { return indices_view(parent_->cache_); }

    template <unsigned N = 0>
    decltype(auto) bin(std::integral_constant<unsigned, N> = {}) const {
      return parent_->hist_.axis(std::integral_constant<unsigned, N>())[index(N)];
    }

    decltype(auto) bin(unsigned d) const { return parent_->hist_.axis(d)[index(d)]; }

    double density() const {
      double x = 1;
      auto it = parent_->cache_.begin();
      parent_->hist_.for_each_axis([&](const auto& a) {
        const auto w = axis::traits::width_as<double>(a, it++->idx);
        x *= w ? w : 1;
      });
      return *iter_ / x;
    }

  private:
    accessor(indexed_range* parent, histogram_iterator i) : parent_(parent), iter_(i) {}
    indexed_range* parent_;
    histogram_iterator iter_;
    friend class indexed_range;
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

  indexed_range(Histogram& h, coverage c)
      : hist_(h)
      , cover_all_(c == coverage::all)
      , begin_(hist_.begin())
      , end_(begin_)
      , cache_(hist_.rank()) {
    auto ca = cache_.begin();
    std::size_t stride = 1;
    h.for_each_axis([&, this](const auto& a) {
      const auto opt = axis::traits::options(a);
      const auto shift = test(opt, axis::option::underflow);

      ca->extend = axis::traits::extend(a);
      ca->begin = cover_all_ ? -shift : 0;
      ca->end = ca->extend - shift - (cover_all_ ? 0 : test(opt, axis::option::overflow));
      ca->idx = ca->begin;

      begin_ += (ca->begin + shift) * stride;
      if ((ca + 1) < cache_.end())
        end_ += (ca->begin + shift) * stride;
      else
        end_ += (ca->end + shift) * stride;

      stride *= ca->extend;
      ++ca;
    });
  }

  range_iterator begin() noexcept { return {this, begin_}; }
  range_iterator end() noexcept { return {this, end_}; }

private:
  Histogram& hist_;
  const bool cover_all_;
  histogram_iterator begin_, end_;
  mutable cache_type cache_;
}; // namespace histogram

template <typename Histogram>
indexed_range<std::remove_reference_t<Histogram>> indexed(
    Histogram&& h, coverage c = coverage::use_default) {
  return {std::forward<Histogram>(h), c};
}

} // namespace histogram
} // namespace boost

#endif
