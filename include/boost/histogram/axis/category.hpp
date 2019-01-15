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
#include <boost/container/vector.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/compressed_pair.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

template <class, class, bool>
class optional_category_mixin {};

/** Axis which maps unique values to bins (one on one).
 *
 * The axis maps a set of values to bins, following the order of
 * arguments in the constructor. There is an optional overflow bin
 * for this axis, which counts values that are not part of the set.
 * Binning has a O(N) complexity, but with a very small factor. For
 * small N (the typical use case) it beats other kinds of lookup.
 * Value types must be equal-omparable.
 */
template <class T, class MetaData, option Options, class Allocator>
class category : public iterator_mixin<category<T, MetaData, Options, Allocator>>,
                 public optional_category_mixin<category<T, MetaData, Options, Allocator>,
                                                T, test(Options, option::growth)> {
  static_assert(!test(Options, option::underflow), "category axis cannot have underflow");
  static_assert(!test(Options, option::circular), "category axis cannot be circular");
  using metadata_type = MetaData;
  using value_type = T;
  using allocator_type = Allocator;
  using vector_type = boost::container::vector<value_type, allocator_type>;

public:
  category() = default;

  /** Construct from iterator range of unique values.
   *
   * \param begin     begin of category range of unique values.
   * \param end       end of category range of unique values.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <class It, class = detail::requires_iterator<It>>
  category(It begin, It end, metadata_type m = {}, allocator_type a = {})
      : vec_meta_(vector_type(begin, end, a), std::move(m)) {
    if (size() == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("bins > 0 required"));
  }

  /** Construct axis from iterable sequence of unique values.
   *
   * \param seq sequence of unique values.
   * \param metadata description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <class C, class = detail::requires_iterable<C>>
  category(const C& iterable, metadata_type m = {}, allocator_type a = {})
      : category(std::begin(iterable), std::end(iterable), std::move(m), std::move(a)) {}

  /** Construct axis from an initializer list of unique values.
   *
   * \param seq sequence of unique values.
   * \param metadata description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <class U>
  category(std::initializer_list<U> l, metadata_type m = {}, allocator_type a = {})
      : category(l.begin(), l.end(), std::move(m), std::move(a)) {}

  /// Returns the bin index for the passed argument.
  int operator()(const value_type& x) const noexcept {
    const auto beg = vec_meta_.first().begin();
    const auto end = vec_meta_.first().end();
    return std::distance(beg, std::find(beg, end, x));
  }

  /// Returns the value for the bin index (performs a range check).
  decltype(auto) value(int idx) const {
    if (idx < 0 || idx >= size())
      BOOST_THROW_EXCEPTION(std::out_of_range("category index out of range"));
    return vec_meta_.first()[idx];
  }

  decltype(auto) operator[](int idx) const noexcept { return value(idx); }

  /// Returns the number of bins, without extra bins.
  int size() const noexcept { return vec_meta_.first().size(); }
  /// Returns the options.
  static constexpr option options() noexcept { return Options; }
  /// Returns the metadata.
  metadata_type& metadata() noexcept { return vec_meta_.second(); }
  /// Returns the metadata (const version).
  const metadata_type& metadata() const noexcept { return vec_meta_.second(); }
  bool operator==(const category& o) const noexcept {
    const auto& a = vec_meta_.first();
    const auto& b = o.vec_meta_.first();
    return std::equal(a.begin(), a.end(), b.begin(), b.end()) &&
           detail::relaxed_equal(metadata(), o.metadata());
  }
  bool operator!=(const category& o) const noexcept { return !operator==(o); }

  allocator_type get_allocator() const { return vec_meta_.first().get_allocator(); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  detail::compressed_pair<vector_type, metadata_type> vec_meta_;
  template <class, class, bool>
  friend class optional_category_mixin;
};

template <class Derived, class T>
class optional_category_mixin<Derived, T, true> {
  using value_type = T;

public:
  auto update(const value_type& x) {
    auto& der = *static_cast<Derived*>(this);
    const auto i = der(x);
    if (i < der.size()) return std::make_pair(i, 0);
    der.vec_meta_.first().emplace_back(x);
    return std::make_pair(i, -1);
  }
};

#if __cpp_deduction_guides >= 201606

template <class T>
category(std::initializer_list<T>)->category<T>;

category(std::initializer_list<const char*>)->category<boost::container::string>;

template <class T>
category(std::initializer_list<T>, const char*)->category<T>;

template <class T, class M>
category(std::initializer_list<T>, const M&)->category<T, M>;

#endif

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
