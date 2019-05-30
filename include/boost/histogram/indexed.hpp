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
#include <boost/histogram/fwd.hpp>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

/** Coverage mode of the indexed range generator.

  Defines options for the iteration strategy.
*/
enum class coverage {
  inner, /*!< iterate over inner bins, exclude underflow and overflow */
  all,   /*!< iterate over all bins, including underflow and overflow */
};

/** Input iterator range over histogram bins with multi-dimensional index.

  The iterator returned by begin() can only be incremented. begin() may only be called
  once, calling it a second time returns the end() iterator. If several copies of the
  input iterators exist, the other copies become invalid if one of them is incremented.
*/
template <class Histogram>
class BOOST_HISTOGRAM_NODISCARD indexed_range {
private:
  using histogram_type = Histogram;
  static constexpr std::size_t buffer_size =
      detail::buffer_size<typename std::decay_t<histogram_type>::axes_type>::value;

public:
  using value_iterator = decltype(std::declval<Histogram>().begin());
  using value_reference = typename value_iterator::reference;
  class range_iterator;

private:
  struct state_type {
    struct index_data {
      axis::index_type idx, begin, end, extent;
    };

    state_type(histogram_type& h) : hist_(h) {}

    histogram_type& hist_;
    index_data indices_[buffer_size];
  };

public:
  class detached_accessor; // forward declaration

  /** Pointer-like class to access value and index of current cell.

    Its methods allow one to query the current indices and bins. Furthermore, it acts
    like a pointer to the cell value. The accessor is coupled to the current
    range_iterator. Moving the range_iterator forward invalidates the accessor. Use the
    detached_accessor class if you must store accessors for later use, but be aware
    that a detached_accessor has a state many times larger than a pointer.
  */
  class accessor {
  public:
    /// Array-like view into the current multi-dimensional index.
    class index_view {
      using index_pointer = const typename state_type::index_data*;

    public:
      using reference = const axis::index_type&;

      /// implementation detail
      class const_iterator
          : public detail::iterator_adaptor<const_iterator, index_pointer, reference> {
      public:
        reference operator*() const noexcept { return const_iterator::base()->idx; }

      private:
        explicit const_iterator(index_pointer i) noexcept
            : const_iterator::iterator_adaptor_(i) {}
        friend class index_view;
      };

      const_iterator begin() const noexcept { return const_iterator{begin_}; }
      const_iterator end() const noexcept { return const_iterator{end_}; }
      std::size_t size() const noexcept {
        return static_cast<std::size_t>(end_ - begin_);
      }
      reference operator[](unsigned d) const noexcept { return begin_[d].idx; }
      reference at(unsigned d) const { return begin_[d].idx; }

    private:
      /// implementation detail
      index_view(index_pointer b, index_pointer e) : begin_(b), end_(e) {}

      index_pointer begin_, end_;
      friend class accessor;
    };

    /// Returns the cell reference.
    value_reference get() const noexcept { return *iter_; }
    /// @copydoc get()
    value_reference operator*() const noexcept { return get(); }
    /// Access fields and methods of the cell object.
    value_iterator operator->() const noexcept { return iter_; }

    /// Access current index.
    /// @param d axis dimension.
    axis::index_type index(unsigned d = 0) const noexcept {
      return state_.indices_[d].idx;
    }

    /// Access indices as an iterable range.
    index_view indices() const noexcept {
      return {state_.indices_, state_.indices_ + state_.hist_.rank()};
    }

    /// Access current bin.
    /// @tparam N axis dimension.
    template <unsigned N = 0>
    decltype(auto) bin(std::integral_constant<unsigned, N> = {}) const {
      return state_.hist_.axis(std::integral_constant<unsigned, N>()).bin(index(N));
    }

    /// Access current bin.
    /// @param d axis dimension.
    decltype(auto) bin(unsigned d) const { return state_.hist_.axis(d).bin(index(d)); }

    /** Computes density in current cell.

      The density is computed as the cell value divided by the product of bin widths. Axes
      without bin widths, like axis::category, are treated as having unit bin with.
    */
    double density() const {
      double x = 1;
      unsigned d = 0;
      state_.hist_.for_each_axis([&](const auto& a) {
        const auto w = axis::traits::width_as<double>(a, this->index(d++));
        x *= w ? w : 1;
      });
      return get() / x;
    }

  protected:
    accessor(state_type& s, value_iterator i) noexcept : state_(s), iter_(i) {}

    state_type& state_;
    value_iterator iter_;
    friend class range_iterator;
    friend class detached_accessor;
  };

  /// Accessor that owns a copy of the iterator state.
  class detached_accessor : public accessor {
  public:
    detached_accessor(const accessor& x) : accessor(state_, x.iter_), state_(x.state_) {}
    detached_accessor(const detached_accessor& x)
        : detached_accessor(static_cast<const accessor&>(x)) {}
    detached_accessor& operator=(const detached_accessor& x) {
      state_ = x.state_;
      accessor::iter_ = x.iter_;
      return *this;
    }

  private:
    state_type state_;
  };

  /// implementation detail
  class range_iterator {
  public:
    using value_type = accessor;
    using reference = accessor&;
    using pointer = accessor*;
    using difference_type = void;
    using iterator_category = std::input_iterator_tag;

  private:
    struct value_proxy {
      detached_accessor operator*() { return ref; }
      detached_accessor ref;
    };

  public:
    reference operator*() noexcept { return value_; }
    pointer operator->() noexcept { return &value_; }

    range_iterator& operator++() {
      std::size_t stride = 1;
      auto c = value_.state_.indices_;
      ++c->idx;
      ++value_.iter_;
      while (c->idx == c->end &&
             (c != (value_.state_.indices_ + value_.state_.hist_.rank() - 1))) {
        c->idx = c->begin;
        value_.iter_ -= (c->end - c->begin) * stride;
        stride *= c->extent;
        ++c;
        ++c->idx;
        value_.iter_ += stride;
      }
      return *this;
    }

    value_proxy operator++(int) {
      value_proxy x{value_};
      operator++();
      return x;
    }

    bool operator==(const range_iterator& x) const noexcept {
      return value_.iter_ == x.value_.iter_;
    }
    bool operator!=(const range_iterator& x) const noexcept { return !operator==(x); }

  private:
    range_iterator(state_type& s, value_iterator i) noexcept : value_(s, i) {}

    accessor value_;
    friend class indexed_range;
  };

  indexed_range(Histogram& hist, coverage cov)
      : state_(hist), begin_(hist.begin()), end_(begin_) {
    std::size_t stride = 1;
    auto ca = state_.indices_;
    const auto clast = ca + state_.hist_.rank() - 1;
    state_.hist_.for_each_axis([ca, clast, cov, &stride, this](const auto& a) mutable {
      using opt = axis::traits::static_options<decltype(a)>;
      constexpr int under = opt::test(axis::option::underflow);
      constexpr int over = opt::test(axis::option::overflow);
      const auto size = a.size();

      ca->extent = size + under + over;
      // -1 if underflow and cover all, else 0
      ca->begin = cov == coverage::all ? -under : 0;
      // size + 1 if overflow and cover all, else size
      ca->end = cov == coverage::all ? size + over : size;
      ca->idx = ca->begin;

      begin_ += (ca->begin + under) * stride;
      end_ += ((ca < clast ? ca->begin : ca->end) + under) * stride;

      stride *= ca->extent;
      ++ca;
    });
  }

  range_iterator begin() noexcept {
    auto begin = begin_;
    begin_ = end_;
    return {state_, begin};
  }
  range_iterator end() noexcept { return {state_, end_}; }

private:
  state_type state_;
  value_iterator begin_, end_;
};

/** Generates an indexed iterator range over the histogram cells.

  Use this in a range-based for loop:

  ```
  for (auto x : indexed(hist)) { ... }
  ```

  This highly optimized loop is at least comparable in speed to a hand-written loop over
  the histogram cells and often much faster, depending on the histogram configuration. The
  iterators dereference to an indexed_range::accessor, which has methods to query the
  current indices and bins and acts like a pointer to the cell value. Accessors, like
  pointers, are cheap to copy but get invalidated when the range iterator is incremented.
  Likewise, any copies of a range iterator become invalid if one of them is incremented.
  A indexed_range::detached_accessor can be stored for later use, but manually copying the
  data of interest from the accessor is usually more efficient.

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
