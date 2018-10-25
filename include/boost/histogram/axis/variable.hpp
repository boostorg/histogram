// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_VARIABLE_HPP
#define BOOST_HISTOGRAM_AXIS_VARIABLE_HPP

#include <algorithm>
#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/buffer.hpp>
#include <boost/histogram/detail/compressed_pair.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace axis {
/** Axis for non-equidistant bins on the real line.
 *
 * Binning is a O(log(N)) operation. If speed matters and the problem
 * domain allows it, prefer a regular axis, possibly with a transform.
 */
template <typename RealType, typename Allocator, typename MetaData>
class variable : public base<MetaData>,
                 public iterator_mixin<variable<RealType, Allocator, MetaData>> {
  using base_type = base<MetaData>;
  using allocator_type = Allocator;
  using metadata_type = MetaData;
  using value_type = RealType;

public:
  /** Construct from iterator range of bin edges.
   *
   * \param begin     begin of edge sequence.
   * \param end       end of edge sequence.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <typename It, typename = detail::requires_iterator<It>>
  variable(It begin, It end, metadata_type m = metadata_type(),
           option_type o = option_type::underflow_and_overflow,
           allocator_type a = allocator_type())
      : base_type(begin == end ? 0 : std::distance(begin, end) - 1, std::move(m), o)
      , x_(nullptr, std::move(a)) {
    using AT = std::allocator_traits<allocator_type>;
    x_.first() = AT::allocate(x_.second(), nx());
    try {
      auto xit = x_.first();
      try {
        AT::construct(x_.second(), xit, *begin++);
        while (begin != end) {
          if (*begin <= *xit) {
            ++xit; // to make sure catch code works
            throw std::invalid_argument("input sequence must be strictly ascending");
          }
          ++xit;
          AT::construct(x_.second(), xit, *begin++);
        }
      } catch (...) {
        // release resources that were already acquired before rethrowing
        while (xit != x_.first()) AT::destroy(x_.second(), --xit);
        throw;
      }
    } catch (...) {
      // release resources that were already acquired before rethrowing
      AT::deallocate(x_.second(), x_.first(), nx());
      throw;
    }
  }

  /** Construct variable axis from iterable range of bin edges.
   *
   * \param iterable  iterable range of bin edges.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <typename U, typename = detail::requires_iterable<U>>
  variable(const U& iterable, metadata_type m = metadata_type(),
           option_type o = option_type::underflow_and_overflow,
           allocator_type a = allocator_type())
      : variable(std::begin(iterable), std::end(iterable), std::move(m), o,
                 std::move(a)) {}

  /** Construct variable axis from initializer list of bin edges.
   *
   * \param edgelist  list of of bin edges.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <typename U>
  variable(const std::initializer_list<U>& l, metadata_type m = metadata_type(),
           option_type o = option_type::underflow_and_overflow,
           allocator_type a = allocator_type())
      : variable(l.begin(), l.end(), std::move(m), o, std::move(a)) {}

  variable() : x_(nullptr) {}

  variable(const variable& o) : base_type(o), x_(o.x_) {
    x_.first() = detail::create_buffer_from_iter(x_.second(), nx(), o.x_.first());
  }

  variable& operator=(const variable& o) {
    if (this != &o) {
      if (base_type::size() == o.size()) {
        base_type::operator=(o);
        std::copy(o.x_.first(), o.x_.first() + nx(), x_.first());
      } else {
        detail::destroy_buffer(x_.second(), x_.first(), nx());
        base_type::operator=(o);
        x_.second() = o.x_.second();
        x_.first() = detail::create_buffer_from_iter(x_.second(), nx(), o.x_.first());
      }
    }
    return *this;
  }

  variable(variable&& o) : variable() {
    using std::swap;
    swap(static_cast<base_type&>(*this), static_cast<base_type&>(o));
    swap(x_, o.x_);
  }

  variable& operator=(variable&& o) {
    if (this != &o) {
      using std::swap;
      swap(static_cast<base_type&>(*this), static_cast<base_type&>(o));
      swap(x_, o.x_);
    }
    return *this;
  }

  ~variable() { detail::destroy_buffer(x_.second(), x_.first(), nx()); }

  /// Returns the bin index for the passed argument.
  int operator()(value_type x) const noexcept {
    const auto p = x_.first();
    return std::upper_bound(p, p + nx(), x) - p - 1;
  }

  /// Returns axis value for fractional index.
  value_type value(value_type i) const noexcept {
    if (i < 0) { return -std::numeric_limits<value_type>::infinity(); }
    if (i > base_type::size()) { return std::numeric_limits<value_type>::infinity(); }
    value_type z;
    const int k = std::modf(i, &z);
    return (1.0 - z) * x_.first()[k] + z * x_.first()[k + 1];
  }

  auto operator[](int idx) const noexcept { return interval_view<variable>(idx, *this); }

  bool operator==(const variable& o) const noexcept {
    return base_type::operator==(o) &&
           std::equal(x_.first(), x_.first() + nx(), o.x_.first());
  }

  bool operator!=(const variable<>& o) const noexcept { return !operator==(o); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  int nx() const { return base_type::size() + 1; }
  using pointer = typename std::allocator_traits<allocator_type>::pointer;
  detail::compressed_pair<pointer, allocator_type> x_;
};
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
