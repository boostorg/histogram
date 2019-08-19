// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_LINEARIZE_HPP
#define BOOST_HISTOGRAM_DETAIL_LINEARIZE_HPP

#include <boost/assert.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/optional_index.hpp>
#include <boost/histogram/fwd.hpp>

namespace boost {
namespace histogram {
namespace detail {

// no underflow, no overflow
inline void linearize(mp11::mp_false, mp11::mp_false, optional_index& out,
                      const std::size_t stride, const axis::index_type size,
                      const axis::index_type i) noexcept {
  if (i >= 0 && i < size)
    out += i * stride;
  else
    out = optional_index::invalid;
}

// no underflow, overflow
inline void linearize(mp11::mp_false, mp11::mp_true, optional_index& out,
                      const std::size_t stride, const axis::index_type size,
                      const axis::index_type i) noexcept {
  BOOST_ASSERT(i <= size);
  if (i >= 0)
    out += i * stride;
  else
    out = optional_index::invalid;
}

// underflow, no overflow
inline void linearize(mp11::mp_true, mp11::mp_false, optional_index& out,
                      const std::size_t stride, const axis::index_type size,
                      const axis::index_type i) noexcept {
  // internal index must be shifted by + 1 since axis has underflow bin
  BOOST_ASSERT(i + 1 >= 0);
  if (i < size)
    out += (i + 1) * stride;
  else
    out = optional_index::invalid;
}

// underflow, overflow
inline void linearize(mp11::mp_true, mp11::mp_true, optional_index& out,
                      const std::size_t stride, const axis::index_type size,
                      const axis::index_type i) noexcept {
  // internal index must be shifted by +1 since axis has underflow bin
  BOOST_ASSERT(i + 1 >= 0);
  BOOST_ASSERT(i < size + 1);
  out += (i + 1) * stride;
}

template <class HasUnderflow, class HasOverflow>
inline void linearize(HasUnderflow, HasOverflow, std::size_t& out,
                      const std::size_t stride, const axis::index_type size,
                      const axis::index_type i) noexcept {
  // internal index must be shifted by +1 if axis has underflow bin
  BOOST_ASSERT((HasUnderflow::value ? i + 1 : i) >= 0);
  BOOST_ASSERT(i < (HasOverflow::value ? size + 1 : size));
  out += (HasUnderflow::value ? i + 1 : i) * stride;
}

template <class Axis, class Value>
std::size_t linearize(optional_index& o, const std::size_t s, const Axis& a,
                      const Value& v) {
  using O = axis::traits::static_options<Axis>;
  linearize(O::test(axis::option::underflow), O::test(axis::option::overflow), o, s,
            a.size(), axis::traits::index(a, v));
  return axis::traits::extent(a);
}

template <class Axis, class Value>
std::size_t linearize(std::size_t& o, const std::size_t s, const Axis& a,
                      const Value& v) {
  using O = axis::traits::static_options<Axis>;
  linearize(O::test(axis::option::underflow), O::test(axis::option::overflow), o, s,
            a.size(), axis::traits::index(a, v));
  return axis::traits::extent(a);
}

template <class... Ts, class Value>
std::size_t linearize(optional_index& o, const std::size_t s,
                      const axis::variant<Ts...>& a, const Value& v) {
  return axis::visit([&o, &s, &v](const auto& a) { return linearize(o, s, a, v); }, a);
}

template <class... Ts, class Value>
std::size_t linearize(std::size_t& o, const std::size_t s, const axis::variant<Ts...>& a,
                      const Value& v) {
  return axis::visit([&o, &s, &v](const auto& a) { return linearize(o, s, a, v); }, a);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif // BOOST_HISTOGRAM_DETAIL_LINEARIZE_HPP
