// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_INTEGER_HPP
#define BOOST_HISTOGRAM_AXIS_INTEGER_HPP

#include <boost/container/string.hpp> // default meta data
#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {
/** Axis for an interval of integer values with unit steps.
 *
 * Binning is a O(1) operation. This axis operates
 * faster than a regular axis.
 */
template <typename IntType, typename MetaData, option Options>
class integer : public base<MetaData, Options>,
                public iterator_mixin<integer<IntType, MetaData, Options>> {
  static_assert(!test(Options, option::circular) || !test(Options, option::underflow),
                "circular axis cannot have underflow");
  static_assert(!std::is_integral<IntType>::value || std::is_same<IntType, int>::value,
                "integer axis requires type floating point type or int");
  using base_type = base<MetaData, Options>;

public:
  using metadata_type = MetaData;
  using value_type = IntType;

private:
  using index_type = std::conditional_t<std::is_integral<value_type>::value, int, double>;

public:
  integer() = default;

  /** Construct over semi-open integer interval [start, stop).
   *
   * \param start    first integer of covered range.
   * \param stop     one past last integer of covered range.
   * \param metadata description of the axis.
   * \param options  extra bin options.
   */
  integer(value_type start, value_type stop, metadata_type m = {})
      : base_type(static_cast<unsigned>(stop - start > 0 ? stop - start : 0),
                  std::move(m))
      , min_(start) {}

  /// Constructor used by algorithm::reduce to shrink and rebin.
  integer(const integer& src, int begin, int end, unsigned merge)
      : base_type(end - begin, src.metadata()), min_(src.min_ + begin) {
    if (merge > 1)
      BOOST_THROW_EXCEPTION(std::invalid_argument("cannot merge bins for integer axis"));
    if (test(Options, option::circular) && !(begin == 0 && end == src.size()))
      BOOST_THROW_EXCEPTION(std::invalid_argument("cannot shrink circular axis"));
  }

  /// Returns the bin index for the passed argument.
  int operator()(value_type x) const noexcept {
    return index_impl(std::is_floating_point<value_type>(), x);
  }

  template <option O = Options, class = std::enable_if_t<test(O, option::growth)>>
  auto update(value_type x) {
    return update_impl(std::is_floating_point<value_type>(), x);
  }

  /// Returns axis value for index.
  value_type value(index_type i) const noexcept {
    if (!test(Options, option::circular)) {
      if (i < 0) return detail::lowest<value_type>();
      if (i > base_type::size()) { return detail::highest<value_type>(); }
    }
    return min_ + i;
  }

  decltype(auto) operator[](int idx) const noexcept {
    return subscript_impl(std::is_same<index_type, double>(), idx);
  }

  bool operator==(const integer& o) const noexcept {
    return base_type::operator==(o) && min_ == o.min_;
  }

  bool operator!=(const integer& o) const noexcept { return !operator==(o); }

  template <class Archive>
  void serialize(Archive&, unsigned);

protected:
  int index_impl(std::false_type, int x) const noexcept {
    const auto z = x - min_;
    if (test(Options, option::circular))
      return z - std::floor(float(z) / base_type::size()) * base_type::size();
    if (z < base_type::size()) return z >= 0 ? z : -1;
    return base_type::size();
  }

  template <typename T>
  int index_impl(std::true_type, T x) const noexcept {
    // need to handle NaN, cannot simply cast to int and call int-implementation
    const auto z = std::floor(x - min_);
    if (test(Options, option::circular)) {
      if (std::isfinite(z))
        return static_cast<int>(z -
                                std::floor(z / base_type::size()) * base_type::size());
    } else if (z < base_type::size()) {
      return z >= 0 ? static_cast<int>(z) : -1;
    }
    return base_type::size();
  }

  auto update_impl(std::false_type, int x) noexcept {
    const auto i = x - min_;
    if (i >= 0) {
      if (i < base_type::size()) return std::make_pair(i, 0);
      const auto n = i - base_type::size() + 1;
      base_type::grow(n);
      return std::make_pair(i, -n);
    }
    min_ += i;
    base_type::grow(-i);
    return std::make_pair(0, -i);
  }

  template <class T>
  auto update_impl(std::true_type, T x) {
    if (std::isfinite(x))
      return update_impl(std::false_type{}, static_cast<int>(std::floor(x)));
    BOOST_THROW_EXCEPTION(std::invalid_argument("argument is not finite"));
    return std::make_pair(0, 0);
  }

  decltype(auto) subscript_impl(std::true_type, int idx) const noexcept {
    return interval_view<integer>(*this, idx);
  }

  decltype(auto) subscript_impl(std::false_type, int idx) const noexcept {
    return value(idx);
  }

  int min_ = 0;
};

#if __cpp_deduction_guides >= 201606

template <class T>
integer(T, T)->integer<detail::convert_integer<T, int>>;

template <class T>
integer(T, T, const char*)->integer<detail::convert_integer<T, int>>;

template <class T, class M>
integer(T, T, M)->integer<detail::convert_integer<T, int>, M>;

#endif

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
