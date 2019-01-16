// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_INTEGER_HPP
#define BOOST_HISTOGRAM_AXIS_INTEGER_HPP

#include <boost/container/string.hpp> // default meta data
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/compressed_pair.hpp>
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

template <class, class, bool>
class optional_integer_mixin {};

/** Axis for an interval of integer values with unit steps.
 *
 * Binning is a O(1) operation. This axis operates
 * faster than a regular axis.
 */
template <class Value, class MetaData, option Options>
class integer : public iterator_mixin<integer<Value, MetaData, Options>>,
                public optional_integer_mixin<integer<Value, MetaData, Options>, Value,
                                              test(Options, option::growth)> {
  static_assert(!test(Options, option::circular) || !test(Options, option::underflow),
                "circular axis cannot have underflow");
  static_assert(!std::is_integral<Value>::value || std::is_same<Value, index_type>::value,
                "integer axis requires type floating point type or index_type");
  using metadata_type = MetaData;
  using value_type = Value;
  using local_index_type = std::conditional_t<std::is_integral<value_type>::value,
                                              index_type, real_index_type>;

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
      : size_meta_(static_cast<index_type>(stop - start), std::move(m)), min_(start) {
    if (stop <= start) BOOST_THROW_EXCEPTION(std::invalid_argument("bins > 0 required"));
  }

  /// Constructor used by algorithm::reduce to shrink and rebin.
  integer(const integer& src, index_type begin, index_type end, unsigned merge)
      : integer(src.value(begin), src.value(end), src.metadata()) {
    if (merge > 1)
      BOOST_THROW_EXCEPTION(std::invalid_argument("cannot merge bins for integer axis"));
    if (test(Options, option::circular) && !(begin == 0 && end == src.size()))
      BOOST_THROW_EXCEPTION(std::invalid_argument("cannot shrink circular axis"));
  }

  /// Returns the bin index for the passed argument.
  index_type operator()(value_type x) const noexcept {
    return index_impl(std::is_floating_point<value_type>(), x);
  }

  /// Returns axis value for index.
  value_type value(local_index_type i) const noexcept {
    if (!test(Options, option::circular)) {
      if (i < 0) return detail::lowest<value_type>();
      if (i > size()) { return detail::highest<value_type>(); }
    }
    return min_ + i;
  }

  decltype(auto) operator[](local_index_type idx) const noexcept {
    return detail::static_if<std::is_floating_point<local_index_type>>(
        [this](auto idx) { return interval_view<integer>(*this, idx); },
        [this](auto idx) { return value(idx); }, idx);
  }

  /// Returns the number of bins, without extra bins.
  int size() const noexcept { return size_meta_.first(); }
  /// Returns the options.
  static constexpr option options() noexcept { return Options; }
  /// Returns the metadata.
  metadata_type& metadata() noexcept { return size_meta_.second(); }
  /// Returns the metadata (const version).
  const metadata_type& metadata() const noexcept { return size_meta_.second(); }

  bool operator==(const integer& o) const noexcept {
    return size() == o.size() && detail::relaxed_equal(metadata(), o.metadata()) &&
           min_ == o.min_;
  }

  bool operator!=(const integer& o) const noexcept { return !operator==(o); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  index_type index_impl(std::false_type, index_type x) const noexcept {
    const auto z = x - min_;
    if (test(Options, option::circular))
      return z - std::floor(float(z) / size()) * size();
    if (z < size()) return z >= 0 ? z : -1;
    return size();
  }

  template <typename T>
  index_type index_impl(std::true_type, T x) const noexcept {
    // need to handle NaN, cannot simply cast to int and call int-implementation
    const auto z = x - min_;
    if (test(Options, option::circular)) {
      if (std::isfinite(z))
        return static_cast<index_type>(std::floor(z) - std::floor(z / size()) * size());
    } else if (z < size()) {
      return z >= 0 ? static_cast<index_type>(z) : -1;
    }
    return size();
  }

  detail::compressed_pair<index_type, metadata_type> size_meta_{0};
  index_type min_{0};

  template <class, class, bool>
  friend class optional_integer_mixin;
};

template <class Derived, class Value>
class optional_integer_mixin<Derived, Value, true> {
  using value_type = Value;

public:
  /// Returns index and shift (if axis has grown) for the passed argument.
  auto update(value_type x) {
    auto impl = [](auto& der, index_type x) {
      const auto i = x - der.min_;
      if (i >= 0) {
        if (i < der.size()) return std::make_pair(i, 0);
        const auto n = i - der.size() + 1;
        der.size_meta_.first() += n;
        return std::make_pair(i, -n);
      }
      der.min_ += i;
      der.size_meta_.first() -= i;
      return std::make_pair(0, -i);
    };

    return detail::static_if<std::is_floating_point<value_type>>(
        [impl](auto& der, auto x) {
          if (std::isfinite(x)) return impl(der, static_cast<index_type>(std::floor(x)));
          BOOST_THROW_EXCEPTION(std::invalid_argument("argument is not finite"));
          return std::make_pair(0, 0);
        },
        impl, *static_cast<Derived*>(this), x);
  }
};

#if __cpp_deduction_guides >= 201606

template <class T>
integer(T, T)->integer<detail::convert_integer<T, index_type>>;

template <class T>
integer(T, T, const char*)->integer<detail::convert_integer<T, index_type>>;

template <class T, class M>
integer(T, T, M)->integer<detail::convert_integer<T, index_type>, M>;

#endif

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
