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
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/mp11.hpp>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

/// Range over histogram bins with multi-dimensional index.
template <typename Histogram>
class BOOST_HISTOGRAM_NODISCARD indexed_range {
  using value_type = typename Histogram::value_type;
  using value_iterator = typename Histogram::const_iterator;
  struct cache_item {
    int idx, begin, end, extend;
  };
  using cache_type = detail::axes_buffer<typename Histogram::axes_type, cache_item>;

public:
  class accessor {
  public:
    class const_iterator
        : public boost::iterator_adaptor<const_iterator,
                                         typename cache_type::const_iterator> {
    public:
      const_iterator(typename cache_type::const_iterator i)
          : const_iterator::iterator_adaptor_(i) {}
      decltype(auto) operator*() const noexcept {
        return const_iterator::base_reference()->idx;
      }

    private:
      friend class boost::iterator_core_access;
    };

    int operator[](unsigned d) const { return parent_.cache_[d].idx; }
    auto begin() const { return const_iterator(parent_.cache_.begin()); }
    auto end() const { return const_iterator(parent_.cache_.end()); }
    auto size() const { return parent_.cache_.size(); }

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

    decltype(auto) operator*() const noexcept { return *iter_; }
    auto operator-> () const noexcept { return iter_; }

  private:
    const indexed_range& parent_;
    value_iterator iter_;
  };

  class const_iterator
      : public boost::iterator_adaptor<const_iterator, value_iterator, accessor,
                                       boost::forward_traversal_tag, accessor> {
  public:
    const_iterator(const indexed_range& p, value_iterator i) noexcept
        : const_iterator::iterator_adaptor_(i), parent_(p) {}

    auto operator*() const noexcept {
      return accessor(parent_, const_iterator::base_reference());
    }

  private:
    friend class boost::iterator_core_access;

    void increment() noexcept {
      std::size_t stride = 1;
      auto c = parent_.cache_.begin();
      ++c->idx;
      ++const_iterator::base_reference();
      while (c->idx == c->end && ((c + 1) != parent_.cache_.end())) {
        c->idx = c->begin;
        const_iterator::base_reference() -= (c->end - c->idx) * stride;
        stride *= c->extend;
        ++c;
        ++c->idx;
        const_iterator::base_reference() += stride;
      }
    }

    const indexed_range& parent_;
  };

  indexed_range(const Histogram& h, bool include_extra_bins)
      : hist_(h)
      , include_extra_bins_(include_extra_bins)
      , begin_(hist_.begin())
      , end_(begin_)
      , cache_(hist_.rank()) {
    auto c = cache_.begin();
    std::size_t stride = 1;
    const auto extra = include_extra_bins_;
    h.for_each_axis([&](const auto& a) {
      const auto opt = axis::traits::options(a);
      const auto shift = opt & axis::option_type::underflow;

      c->extend = axis::traits::extend(a);
      c->begin = extra ? -shift : 0;
      c->end = c->extend - shift - (extra ? 0 : (opt & axis::option_type::overflow));
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

  decltype(auto) begin() const { return const_iterator(*this, begin_); }
  decltype(auto) end() const { return const_iterator(*this, end_); }

private:
  const Histogram& hist_;
  const bool include_extra_bins_;
  value_iterator begin_, end_;
  mutable cache_type cache_;
}; // namespace histogram

template <typename Histogram>
indexed_range<detail::unqual<Histogram>> indexed(Histogram&& h,
                                                 bool include_extra_bins = false) {
  return {std::forward<Histogram>(h), include_extra_bins};
}

} // namespace histogram
} // namespace boost

#endif
