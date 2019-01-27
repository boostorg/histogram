// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_CATEGORY_HPP
#define BOOST_HISTOGRAM_AXIS_CATEGORY_HPP

#include <algorithm>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/compressed_pair.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {
template <class, class, bool>
class category_mixin {};

template <class Derived, class T>
class category_mixin<Derived, T, true> {
  using value_type = T;

public:
  auto update(const value_type& x) {
    auto& der = static_cast<Derived&>(*this);
    const auto i = der.index(x);
    if (i < der.size()) return std::make_pair(i, 0);
    der.vec_meta_.first().emplace_back(x);
    return std::make_pair(i, -1);
  }
};
} // namespace detail

namespace axis {

/** Maps at a set of unique values to bin indices.
 *
 * The axis maps a set of values to bins, following the order of arguments in the
 * constructor. There is an optional overflow bin for this axis, which counts values that
 * are not part of the set. Binning has a O(N) complexity, but with a very small factor.
 * For small N (the typical use case) it beats other kinds of lookup.
 *
 * Value types must be equal-comparable.
 */
template <class Value, class MetaData, option Options, class Allocator>
class category
    : public iterator_mixin<category<Value, MetaData, Options, Allocator>>,
      public detail::category_mixin<category<Value, MetaData, Options, Allocator>, Value,
                                    test(Options, option::growth)> {
  static_assert(!std::is_floating_point<Value>::value,
                "category axis cannot have floating point value type");
  static_assert(!test(Options, option::underflow), "category axis cannot have underflow");
  static_assert(!test(Options, option::circular), "category axis cannot be circular");
  static_assert(!test(Options, option::growth) || !test(Options, option::overflow),
                "growing category axis cannot have overflow");
  using metadata_type = MetaData;
  using value_type = Value;
  using allocator_type = Allocator;
  using vector_type = std::vector<value_type, allocator_type>;

public:
  category() = default;

  /** Construct from iterator range of unique values.
   *
   * \param begin     begin of category range of unique values.
   * \param end       end of category range of unique values.
   * \param meta      description of the axis.
   * \param alloc     allocator instance to use.
   */
  template <class It, class = detail::requires_iterator<It>>
  category(It begin, It end, metadata_type meta = {}, allocator_type alloc = {})
      : vec_meta_(vector_type(begin, end, alloc), std::move(meta)) {
    if (size() == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("bins > 0 required"));
  }

  /** Construct axis from iterable sequence of unique values.
   *
   * \param iterable sequence of unique values.
   * \param meta     description of the axis.
   * \param alloc    allocator instance to use.
   */
  template <class C, class = detail::requires_iterable<C>>
  category(const C& iterable, metadata_type meta = {}, allocator_type alloc = {})
      : category(std::begin(iterable), std::end(iterable), std::move(meta),
                 std::move(alloc)) {}

  /** Construct axis from an initializer list of unique values.
   *
   * \param list   `std::initializer_list` of unique values.
   * \param meta   description of the axis.
   * \param alloc  allocator instance to use.
   */
  template <class U>
  category(std::initializer_list<U> list, metadata_type meta = {},
           allocator_type alloc = {})
      : category(list.begin(), list.end(), std::move(meta), std::move(alloc)) {}

  /// Constructor used by algorithm::reduce to shrink and rebin.
  category(const category& src, index_type begin, index_type end, unsigned merge)
      : category(src.vec_meta_.first().begin() + begin,
                 src.vec_meta_.first().begin() + end, src.metadata()) {
    if (merge > 1)
      BOOST_THROW_EXCEPTION(std::invalid_argument("cannot merge bins for category axis"));
  }

  /// Return index for value argument.
  index_type index(const value_type& x) const noexcept {
    const auto beg = vec_meta_.first().begin();
    const auto end = vec_meta_.first().end();
    return std::distance(beg, std::find(beg, end, x));
  }

  /// Return value for index argument.
  /// Throws `std::out_of_range` if the index is out of bounds.
  decltype(auto) value(index_type idx) const {
    if (idx < 0 || idx >= size())
      BOOST_THROW_EXCEPTION(std::out_of_range("category index out of range"));
    return vec_meta_.first()[idx];
  }

  /// Return value for index argument.
  decltype(auto) bin(index_type idx) const noexcept { return value(idx); }

  /// Returns the number of bins, without over- or underflow.
  index_type size() const noexcept { return vec_meta_.first().size(); }
  /// Returns the options.
  static constexpr option options() noexcept { return Options; }
  /// Returns reference to metadata.
  metadata_type& metadata() noexcept { return vec_meta_.second(); }
  /// Returns reference to const metadata.
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
  friend class detail::category_mixin;
};

#if __cpp_deduction_guides >= 201606

template <class T>
category(std::initializer_list<T>)->category<T>;

category(std::initializer_list<const char*>)->category<std::string>;

template <class T>
category(std::initializer_list<T>, const char*)->category<T>;

template <class T, class M>
category(std::initializer_list<T>, const M&)->category<T, M>;

#endif

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
