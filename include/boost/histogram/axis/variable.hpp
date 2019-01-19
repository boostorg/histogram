// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_VARIABLE_HPP
#define BOOST_HISTOGRAM_AXIS_VARIABLE_HPP

#include <algorithm>
#include <boost/container/new_allocator.hpp>
#include <boost/container/string.hpp> // default meta data
#include <boost/container/vector.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/compressed_pair.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

template <class, class, bool>
class optional_variable_mixin {};

/** Axis for non-equidistant bins on the real line.
 *
 * Binning is a O(log(N)) operation. If speed matters and the problem
 * domain allows it, prefer a regular axis, possibly with a transform.
 */
template <class Value, class MetaData, option Options, class Allocator>
class variable
    : public iterator_mixin<variable<Value, MetaData, Options, Allocator>>,
      public optional_variable_mixin<variable<Value, MetaData, Options, Allocator>, Value,
                                     test(Options, option::growth)> {
  static_assert(!test(Options, option::circular) || !test(Options, option::underflow),
                "circular axis cannot have underflow");
  static_assert(std::is_floating_point<Value>::value,
                "variable axis requires floating point type");

  using metadata_type = MetaData;
  using value_type = Value;
  using allocator_type = Allocator;
  using vec_type = boost::container::vector<Value, allocator_type>;

public:
  variable() = default;

  /** Construct from iterator range of bin edges.
   *
   * \param begin     begin of edge sequence.
   * \param end       end of edge sequence.
   * \param metadata  description of the axis.
   * \param allocator allocator instance to use.
   */
  template <class It, class = detail::requires_iterator<It>>
  variable(It begin, It end, metadata_type m = {}, allocator_type a = {})
      : vec_meta_(vec_type(std::move(a)), std::move(m)) {
    if (std::distance(begin, end) <= 1)
      BOOST_THROW_EXCEPTION(std::invalid_argument("bins > 0 required"));

    auto& v = vec_meta_.first();
    v.reserve(std::distance(begin, end));
    v.emplace_back(*begin++);
    while (begin != end) {
      if (*begin <= v.back())
        BOOST_THROW_EXCEPTION(
            std::invalid_argument("input sequence must be strictly ascending"));
      v.emplace_back(*begin++);
    }
  }

  /** Construct variable axis from iterable range of bin edges.
   *
   * \param iterable  iterable range of bin edges.
   * \param metadata  description of the axis.
   * \param allocator allocator instance to use.
   */
  template <class U, class = detail::requires_iterable<U>>
  variable(const U& iterable, metadata_type m = {}, allocator_type a = {})
      : variable(std::begin(iterable), std::end(iterable), std::move(m), std::move(a)) {}

  /** Construct variable axis from initializer list of bin edges.
   *
   * \param edgelist  list of of bin edges.
   * \param metadata  description of the axis.
   * \param allocator allocator instance to use.
   */
  template <class U>
  variable(std::initializer_list<U> l, metadata_type m = {}, allocator_type a = {})
      : variable(l.begin(), l.end(), std::move(m), std::move(a)) {}

  /// Constructor used by algorithm::reduce to shrink and rebin (not for users).
  variable(const variable& src, index_type begin, index_type end, unsigned merge)
      : vec_meta_(vec_type(src.get_allocator()), src.metadata()) {
    BOOST_ASSERT((end - begin) % merge == 0);
    if (test(Options, option::circular) && !(begin == 0 && end == src.size()))
      BOOST_THROW_EXCEPTION(std::invalid_argument("cannot shrink circular axis"));
    auto& vec = vec_meta_.first();
    vec.reserve((end - begin) / merge);
    const auto beg = src.vec_meta_.first().begin();
    for (index_type i = begin; i <= end; i += merge) vec.emplace_back(*(beg + i));
  }

  /// Returns the bin index for the passed argument.
  int operator()(value_type x) const noexcept {
    const auto& v = vec_meta_.first();
    if (test(Options, option::circular)) {
      const auto a = v[0];
      const auto b = v[size()];
      x -= std::floor((x - a) / (b - a)) * (b - a);
    }
    return std::upper_bound(v.begin(), v.end(), x) - v.begin() - 1;
  }

  /// Returns axis value for fractional index.
  value_type value(real_index_type i) const noexcept {
    const auto& v = vec_meta_.first();
    if (test(Options, option::circular)) {
      auto shift = std::floor(i / size());
      i -= shift * size();
      double z;
      const auto k = static_cast<index_type>(std::modf(i, &z));
      const auto a = v[0];
      const auto b = v[size()];
      return (1.0 - z) * v[k] + z * v[k + 1] + shift * (b - a);
    }
    if (i < 0) return detail::lowest<value_type>();
    if (i == size()) return v.back();
    if (i > size()) return detail::highest<value_type>();
    const auto k = static_cast<index_type>(i); // precond: i >= 0
    const real_index_type z = i - k;
    return (1.0 - z) * v[k] + z * v[k + 1];
  }

  auto operator[](index_type idx) const noexcept {
    return interval_view<variable>(*this, idx);
  }

  /// Returns the number of bins, without extra bins.
  index_type size() const noexcept { return vec_meta_.first().size() - 1; }
  /// Returns the options.
  static constexpr option options() noexcept { return Options; }
  /// Returns the metadata.
  metadata_type& metadata() noexcept { return vec_meta_.second(); }
  /// Returns the metadata (const version).
  const metadata_type& metadata() const noexcept { return vec_meta_.second(); }
  bool operator==(const variable& o) const noexcept {
    const auto& a = vec_meta_.first();
    const auto& b = o.vec_meta_.first();
    return std::equal(a.begin(), a.end(), b.begin(), b.end()) &&
           detail::relaxed_equal(metadata(), o.metadata());
  }
  bool operator!=(const variable<>& o) const noexcept { return !operator==(o); }

  allocator_type get_allocator() const { return vec_meta_.first().get_allocator(); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  detail::compressed_pair<vec_type, metadata_type> vec_meta_;

  template <class, class, bool>
  friend class optional_variable_mixin;
};

template <class Derived, class Value>
class optional_variable_mixin<Derived, Value, true> {
  using value_type = Value;

public:
  auto update(value_type x) noexcept {
    auto& der = static_cast<Derived&>(*this);
    const auto i = der(x);
    if (std::isfinite(x)) {
      auto& vec = der.vec_meta_.first();
      if (0 <= i) {
        if (i < der.size()) return std::make_pair(i, 0);
        const auto d = der.value(der.size()) - der.value(der.size() - 0.5);
        x = std::nextafter(x, std::numeric_limits<value_type>::max());
        x = std::max(x, vec.back() + d);
        vec.push_back(x);
        return std::make_pair(i, -1);
      }
      const auto d = der.value(0.5) - der.value(0);
      x = std::min(x, der.value(0) - d);
      vec.insert(vec.begin(), x);
      return std::make_pair(0, -i);
    }
    return std::make_pair(x < 0 ? -1 : der.size(), 0);
  }
};

#if __cpp_deduction_guides >= 201606

template <class U, class T = detail::convert_integer<U, double>>
variable(std::initializer_list<U>)->variable<T>;

template <class U, class T = detail::convert_integer<U, double>>
variable(std::initializer_list<U>, const char*)->variable<T>;

template <class U, class M, class T = detail::convert_integer<U, double>>
variable(std::initializer_list<U>, M)->variable<T, M>;

template <class Iterable,
          class T = detail::convert_integer<
              detail::naked<decltype(*std::begin(std::declval<Iterable&>()))>, double>>
variable(Iterable)->variable<T>;

template <class Iterable,
          class T = detail::convert_integer<
              detail::naked<decltype(*std::begin(std::declval<Iterable&>()))>, double>>
variable(Iterable, const char*)->variable<T>;

template <class Iterable, class M,
          class T = detail::convert_integer<
              detail::naked<decltype(*std::begin(std::declval<Iterable&>()))>, double>>
variable(Iterable, M)->variable<T, M>;

#endif

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
