// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_CATEGORY_HPP
#define BOOST_HISTOGRAM_AXIS_CATEGORY_HPP

#include <algorithm>
#include <boost/container/new_allocator.hpp>
#include <boost/container/string.hpp> // default meta data
#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/buffer.hpp>
#include <boost/histogram/detail/compressed_pair.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace axis {
/** Axis which maps unique values to bins (one on one).
 *
 * The axis maps a set of values to bins, following the order of
 * arguments in the constructor. There is an optional overflow bin
 * for this axis, which counts values that are not part of the set.
 * Binning is a O(n) operation for n values in the worst case and O(1) in
 * the best case. The value types must be equal-comparable.
 */
template <typename T, typename MetaData, option_type Options, typename Allocator>
class category : public base<MetaData, Options>,
                 public iterator_mixin<category<T, MetaData, Options, Allocator>> {
  static_assert(!(Options & option_type::underflow),
                "category axis cannot have underflow");
  using base_type = base<MetaData, Options>;
  using metadata_type = MetaData;
  using value_type = T;
  using allocator_type = Allocator;

public:
  /** Construct from iterator range of unique values.
   *
   * \param begin     begin of category range of unique values.
   * \param end       end of category range of unique values.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <typename It, typename = detail::requires_iterator<It>>
  category(It begin, It end, metadata_type m = metadata_type(),
           allocator_type a = allocator_type())
      : base_type(std::distance(begin, end), std::move(m)), x_(nullptr, std::move(a)) {
    x_.first() = detail::create_buffer_from_iter(x_.second(), base_type::size(), begin);
  }

  /** Construct axis from iterable sequence of unique values.
   *
   * \param seq sequence of unique values.
   * \param metadata description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <typename C, typename = detail::requires_iterable<C>>
  category(const C& iterable, metadata_type m = metadata_type(),
           allocator_type a = allocator_type())
      : category(std::begin(iterable), std::end(iterable), std::move(m), std::move(a)) {}

  /** Construct axis from an initializer list of unique values.
   *
   * \param seq sequence of unique values.
   * \param metadata description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <typename U>
  category(std::initializer_list<U> l, metadata_type m = metadata_type(),
           allocator_type a = allocator_type())
      : category(l.begin(), l.end(), std::move(m), std::move(a)) {}

  category() : x_(nullptr) {}

  category(const category& o) : base_type(o), x_(o.x_) {
    x_.first() =
        detail::create_buffer_from_iter(x_.second(), base_type::size(), o.x_.first());
  }

  category& operator=(const category& o) {
    if (this != &o) {
      if (base_type::size() != o.size()) {
        detail::destroy_buffer(x_.second(), x_.first(), base_type::size());
        base_type::operator=(o);
        x_ = o.x_;
        x_.first() =
            detail::create_buffer_from_iter(x_.second(), base_type::size(), o.x_.first());
      } else {
        base_type::operator=(o);
        std::copy(o.x_.first(), o.x_.first() + base_type::size(), x_.first());
      }
    }
    return *this;
  }

  category(category&& o) : category() {
    using std::swap;
    swap(static_cast<base_type&>(*this), static_cast<base_type&>(o));
    swap(x_, o.x_);
  }

  category& operator=(category&& o) {
    if (this != &o) {
      using std::swap;
      swap(static_cast<base_type&>(*this), static_cast<base_type&>(o));
      swap(x_, o.x_);
    }
    return *this;
  }

  ~category() { detail::destroy_buffer(x_.second(), x_.first(), base_type::size()); }

  /// Returns the bin index for the passed argument.
  int operator()(const value_type& x) const noexcept {
    const auto begin = x_.first();
    const auto end = begin + base_type::size();
    return std::distance(begin, std::find(begin, end, x));
  }

  /// Returns the value for the bin index (performs a range check).
  decltype(auto) value(int idx) const {
    if (idx < 0 || idx >= base_type::size())
      BOOST_THROW_EXCEPTION(std::out_of_range("category index out of range"));
    return x_.first()[idx];
  }

  decltype(auto) operator[](int idx) const noexcept { return value(idx); }

  bool operator==(const category& o) const noexcept {
    return base_type::operator==(o) &&
           std::equal(x_.first(), x_.first() + base_type::size(), o.x_.first());
  }

  bool operator!=(const category& o) const noexcept { return !operator==(o); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  using pointer = typename std::allocator_traits<allocator_type>::pointer;
  detail::compressed_pair<pointer, allocator_type> x_;
};
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
