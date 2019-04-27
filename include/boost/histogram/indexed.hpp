// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_INDEXED_HPP
#define BOOST_HISTOGRAM_INDEXED_HPP

#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/attribute.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/iterator_adaptor.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

/**
  Coverage mode of the indexed range generator.

  Defines options for the iteration strategy.
*/
enum class coverage {
  inner, /*!< iterate over inner bins, exclude underflow and overflow */
  all,   /*!< iterate over all bins, including underflow and overflow */
};

/// Range over histogram bins with multi-dimensional index.
template <class Histogram>
class BOOST_HISTOGRAM_NODISCARD indexed_range {
private:
  using max_dim = mp11::mp_size_t<
      detail::buffer_size<typename detail::remove_cvref_t<Histogram>::axes_type>::value>;
  struct cache_item {
    axis::index_type idx, begin, end, extent;
  };

public:
  using value_iterator = decltype(std::declval<Histogram>().begin());
  using value_reference = typename value_iterator::reference;

  /**
    Pointer-like class to access value and index of current cell.

    Its methods allow one to query the current indices and bins. Furthermore, it acts
    like a pointer to the cell value.
  */
  class accessor {
  public:
    /// Array-like view into the current multi-dimensional index.
    class index_view {
    public:
      /// implementation detail
      class const_iterator
          : public detail::iterator_adaptor<const_iterator, const cache_item*,
                                            const axis::index_type&> {
      public:
        const axis::index_type& operator*() const noexcept {
          return const_iterator::base()->idx;
        }

      private:
        explicit const_iterator(const cache_item* i) noexcept
            : const_iterator::iterator_adaptor_(i) {}
        friend class accessor;
      };

      const_iterator begin() const noexcept { return const_iterator{begin_}; }
      const_iterator end() const noexcept { return const_iterator{end_}; }
      std::size_t size() const noexcept {
        return static_cast<std::size_t>(end_ - begin_);
      }
      axis::index_type operator[](unsigned d) const noexcept { return begin_[d].idx; }
      axis::index_type at(unsigned d) const { return begin_[d].idx; }

    private:
      /// implementation detail
      index_view(const cache_item* b, const cache_item* e) : begin_(b), end_(e) {}

      const cache_item *begin_, *end_;
      friend class accessor;
    };

    /// Returns the cell value.
    value_reference operator*() const noexcept { return *iter_; }
    /// Returns the cell value.
    value_reference get() const noexcept { return *iter_; }
    /// Access fields and methods of the cell object.
    value_iterator operator->() const noexcept { return iter_; }

    /// Access current index.
    /// @param d axis dimension.
    axis::index_type index(unsigned d = 0) const noexcept {
      return parent_.cache_[d].idx;
    }

    /// Access indices as an iterable range.
    index_view indices() const noexcept {
      return {parent_.cache_, parent_.cache_ + parent_.hist_.rank()};
    }

    /// Access current bin.
    /// @tparam N axis dimension.
    template <unsigned N = 0>
    decltype(auto) bin(std::integral_constant<unsigned, N> = {}) const {
      return parent_.hist_.axis(std::integral_constant<unsigned, N>()).bin(index(N));
    }

    /// Access current bin.
    /// @param d axis dimension.
    decltype(auto) bin(unsigned d) const { return parent_.hist_.axis(d).bin(index(d)); }

    /**
      Computes density in current cell.

      The density is computed as the cell value divided by the product of bin widths. Axes
      without bin widths, like axis::category, are treated as having unit bin with.
    */
    double density() const {
      double x = 1;
      auto it = parent_.cache_;
      parent_.hist_.for_each_axis([&](const auto& a) {
        const auto w = axis::traits::width_as<double>(a, it++->idx);
        x *= w ? w : 1;
      });
      return *iter_ / x;
    }

  private:
    /// implementation detail
    accessor(indexed_range& p, value_iterator i) : parent_(p), iter_(i) {}

    indexed_range& parent_;
    value_iterator iter_;
    friend class indexed_range;
  };

  /// implementation detail
  class range_iterator {
    using detail_pointer = detail::operator_arrow_dispatch_t<accessor>;

  public:
    using value_type = typename value_iterator::value_type;
    using reference = accessor;
    using pointer = typename detail_pointer::result_type;
    using difference_type = void;
    using iterator_category = std::input_iterator_tag;

    accessor operator*() const noexcept { return {*parent_, iter_}; }
    pointer operator->() const noexcept { return detail_pointer::apply(operator*()); }

    range_iterator& operator++() {
      std::size_t stride = 1;
      auto c = parent_->cache_;
      ++c->idx;
      ++iter_;
      while (c->idx == c->end && (c != (parent_->cache_ + parent_->hist_.rank() - 1))) {
        c->idx = c->begin;
        iter_ -= (c->end - c->begin) * stride;
        stride *= c->extent;
        ++c;
        ++c->idx;
        iter_ += stride;
      }
      return *this;
    }

    range_iterator operator++(int) {
      auto tmp = *this;
      operator++();
      return tmp;
    }

    bool operator==(const range_iterator& x) const noexcept { return iter_ == x.iter_; }
    bool operator!=(const range_iterator& x) const noexcept { return !operator==(x); }

  private:
    range_iterator(indexed_range* p, value_iterator i) noexcept : parent_(p), iter_(i) {}

    mutable indexed_range* parent_;
    value_iterator iter_;

    friend class indexed_range;
  };

  indexed_range(Histogram& h, coverage c)
      : hist_(h), cover_all_(c == coverage::all), begin_(hist_.begin()), end_(begin_) {
    auto ca = cache_;
    const auto clast = ca + hist_.rank() - 1;
    std::size_t stride = 1;
    h.for_each_axis([&, this](const auto& a) {
      using opt = axis::traits::static_options<decltype(a)>;
      constexpr int under = opt::test(axis::option::underflow);
      constexpr int over = opt::test(axis::option::overflow);
      const auto size = a.size();

      ca->extent = size + under + over;
      // -1 if underflow and cover all, else 0
      ca->begin = cover_all_ ? -under : 0;
      // size + 1 if overflow and cover all, else size
      ca->end = cover_all_ ? size + over : size;
      ca->idx = ca->begin;

      begin_ += (ca->begin + under) * stride;
      if (ca < clast)
        end_ += (ca->begin + under) * stride;
      else
        end_ += (ca->end + under) * stride;

      stride *= ca->extent;
      ++ca;
    });
  }

  range_iterator begin() noexcept { return {this, begin_}; }
  range_iterator end() noexcept { return {nullptr, end_}; }

private:
  Histogram& hist_;
  const bool cover_all_;
  value_iterator begin_, end_;
  mutable cache_item cache_[max_dim::value];
};

/**
  Generates a range over the histogram entries.

  Use this in a range-based for loop:

  ```
  for (auto x : indexed(hist)) { ... }
  ```

  The iterators dereference to an indexed_range::accessor, which has methods to query the
  current indices and bins, and acts like a pointer to the cell value.

  @returns indexed_range

  @param hist Reference to the histogram.
  @param cov  Iterate over all or only inner bins (optional, default: inner).
 */
template <typename Histogram>
auto indexed(Histogram&& hist, coverage cov = coverage::inner) {
  return indexed_range<std::remove_reference_t<Histogram>>{std::forward<Histogram>(hist),
                                                           cov};
}

} // namespace histogram
} // namespace boost

#endif
