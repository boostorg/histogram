// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_COMPRESSED_PAIR_HPP
#define BOOST_HISTOGRAM_DETAIL_COMPRESSED_PAIR_HPP

#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {
namespace detail {

template <typename T1, typename T2, bool B>
class compressed_pair_impl;

template <typename T1, typename T2>
class compressed_pair_impl<T1, T2, true> : protected T2 {
public:
  using first_type = T1;
  using second_type = T2;

  compressed_pair_impl() = default;
  compressed_pair_impl(compressed_pair_impl&&) = default;
  compressed_pair_impl(const compressed_pair_impl&) = default;
  compressed_pair_impl& operator=(compressed_pair_impl&&) = default;
  compressed_pair_impl& operator=(const compressed_pair_impl&) = default;

  compressed_pair_impl(const first_type& x, const second_type& y)
      : T2(std::move(y)), first_(std::move(x)) {}

  compressed_pair_impl(first_type&& x, second_type&& y)
      : T2(std::move(y)), first_(std::move(x)) {}

  compressed_pair_impl(first_type&& x) : first_(std::move(x)) {}
  compressed_pair_impl(const first_type& x) : first_(x) {}

  first_type& first() { return first_; }
  second_type& second() { return static_cast<second_type&>(*this); }
  const first_type& first() const { return first_; }
  const second_type& second() const { return static_cast<const second_type&>(*this); }

private:
  first_type first_;
};

template <typename T1, typename T2>
class compressed_pair_impl<T1, T2, false> {
public:
  using first_type = T1;
  using second_type = T2;

  compressed_pair_impl() = default;
  compressed_pair_impl(compressed_pair_impl&&) = default;
  compressed_pair_impl(const compressed_pair_impl&) = default;
  compressed_pair_impl& operator=(compressed_pair_impl&&) = default;
  compressed_pair_impl& operator=(const compressed_pair_impl&) = default;

  compressed_pair_impl(first_type&& x, second_type&& y)
      : first_(std::move(x)), second_(std::move(y)) {}

  compressed_pair_impl(const first_type& x, const second_type& y)
      : first_(x), second_(y) {}

  compressed_pair_impl(first_type&& x) : first_(std::move(x)) {}
  compressed_pair_impl(const first_type& x) : first_(x) {}

  first_type& first() { return first_; }
  second_type& second() { return second_; }
  const first_type& first() const { return first_; }
  const second_type& second() const { return second_; }

private:
  first_type first_;
  second_type second_;
};

template <typename... Ts>
void swap(compressed_pair_impl<Ts...>& a, compressed_pair_impl<Ts...>& b) {
  using std::swap;
  swap(a.first(), b.first());
  swap(a.second(), b.second());
}

template <typename T1, typename T2>
using compressed_pair =
    compressed_pair_impl<T1, T2, (!std::is_final<T2>::value && std::is_empty<T2>::value)>;

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
