// Copyright (c) 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_VARIANT_HPP
#define BOOST_HISTOGRAM_DETAIL_VARIANT_HPP

#include <boost/mp11.hpp>
#include <boost/throw_exception.hpp>
#include <iosfwd>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <class T, class U>
T launder_cast(U&& u) {
  return reinterpret_cast<T>(std::forward<U>(u));
}

// Simple C++14 variant that only depends on boost.mp11 and boost.throw_exception.
//
// * No empty state, first type should have a default ctor that never throws;
//   if it throws anyway, the program aborts.
// * All types must have copy ctors and copy assignment.
// * All types must have noexcept move ctors and noexcept move assignment.
//
template <class... Ts>
class variant {
  using default_type = mp11::mp_first<variant>;
  using N = mp11::mp_size<variant>;

public:
  variant() noexcept { init_default(); }

  variant(const variant& x) {
    x.internal_apply([this, &x](auto i) {
      using T = mp11::mp_at_c<variant, i>;
      this->init_i<T>(i, *x.ptr(mp11::mp_identity<T>{}, i));
    });
  }

  variant(variant&& x) noexcept {
    x.internal_apply([this, &x](auto i) {
      using T = mp11::mp_at_c<variant, i>;
      this->init_i<T>(i, std::move(*x.ptr(mp11::mp_identity<T>{}, i)));
    });
  }

  variant& operator=(const variant& x) {
    x.apply([this](const auto& x) { this->operator=(x); });
    return *this;
  }

  variant& operator=(variant&& x) noexcept {
    x.apply([this](auto&& x) { this->operator=(std::move(x)); });
    return *this;
  }

  template <class U, class = std::enable_if_t<mp11::mp_contains<variant, U>::value>>
  explicit variant(U&& x) noexcept {
    static_assert(std::is_rvalue_reference<decltype(x)>::value, "");
    constexpr auto i = find<U>();
    using T = mp11::mp_at_c<variant, i>;
    static_assert(std::is_nothrow_move_constructible<T>::value, "");
    init_i<T>(i, std::move(x));
  }

  template <class U, class = std::enable_if_t<mp11::mp_contains<variant, U>::value>>
  explicit variant(const U& x) {
    constexpr auto i = find<U>();
    using T = mp11::mp_at_c<variant, i>;
    init_i<T>(i, x);
  }

  template <class U, class = std::enable_if_t<mp11::mp_contains<variant, U>::value>>
  variant& operator=(U&& x) noexcept {
    constexpr auto i = find<U>();
    using T = mp11::mp_at_c<variant, i>;
    static_assert(std::is_nothrow_move_constructible<T>::value, "");
    if (i == index_) {
      *ptr(mp11::mp_identity<T>{}, i) = std::move(x);
    } else {
      destroy();
      init_i<T>(i, std::move(x));
    }
    return *this;
  }

  template <class U, class = std::enable_if_t<mp11::mp_contains<variant, U>::value>>
  variant& operator=(const U& x) {
    constexpr auto i = find<U>();
    using T = mp11::mp_at_c<variant, i>;
    if (i == index_) {
      // nothing to do if T::operator= throws
      *ptr(mp11::mp_identity<T>{}, i) = x;
    } else {
      destroy(); // now in invalid state
      try {
        // if this throws, need to return to valid state
        init_i<T>(i, x);
      } catch (...) {
        init_default();
        throw;
      }
    }
    return *this;
  }

  ~variant() { destroy(); }

  template <class U>
  bool operator==(const U& x) const noexcept {
    constexpr auto i = find<U>();
    static_assert(i < N::value, "argument is not a bounded type");
    using T = mp11::mp_at_c<variant, i>;
    return (i == index_) && *ptr(mp11::mp_identity<T>{}, i) == x;
  }

  template <class U>
  bool operator!=(const U& x) const noexcept {
    constexpr auto i = find<U>();
    static_assert(i < N::value, "argument is not a bounded type");
    using T = mp11::mp_at_c<variant, i>;
    return (i != index_) || *ptr(mp11::mp_identity<T>{}, i) != x;
  }

  bool operator==(const variant& x) const noexcept {
    return x.apply([this](const auto& x) { return this->operator==(x); });
  }

  bool operator!=(const variant& x) const noexcept {
    return x.apply([this](const auto& x) { return this->operator!=(x); });
  }

  unsigned index() const noexcept { return index_; }

  template <class T>
  T& get() {
    T* tp = get_if<T>();
    if (!tp) BOOST_THROW_EXCEPTION(std::runtime_error("T is not the held type"));
    return *tp;
  }

  template <class T>
  const T& get() const {
    const T* tp = get_if<T>();
    if (!tp) BOOST_THROW_EXCEPTION(std::runtime_error("T is not the held type"));
    return *tp;
  }

  template <class T>
  T* get_if() noexcept {
    constexpr auto i = mp11::mp_find<variant, T>{};
    return i == index_ ? ptr(mp11::mp_identity<T>{}, i) : nullptr;
  }

  template <class T>
  const T* get_if() const noexcept {
    constexpr auto i = mp11::mp_find<variant, T>{};
    return i == index_ ? ptr(mp11::mp_identity<T>{}, i) : nullptr;
  }

  template <class Functor>
  decltype(auto) apply(Functor&& functor) const {
    return internal_apply([this, &functor](auto i) -> decltype(auto) {
      using T = mp11::mp_at_c<variant, i>;
      return functor(*(this->ptr(mp11::mp_identity<T>{}, i)));
    });
  }

  template <class Functor>
  decltype(auto) apply(Functor&& functor) {
    return internal_apply([this, &functor](auto i) -> decltype(auto) {
      using T = mp11::mp_at_c<variant, i>;
      return functor(*(this->ptr(mp11::mp_identity<T>{}, i)));
    });
  }

private:
  template <class Functor>
  decltype(auto) internal_apply(Functor&& functor) const {
    return mp11::mp_with_index<sizeof...(Ts)>(index_, functor);
  }

  template <class T, std::size_t N>
  T* ptr(mp11::mp_identity<T>, mp11::mp_size_t<N>) noexcept {
    return launder_cast<T*>(&buffer_);
  }

  template <class T, std::size_t N>
  const T* ptr(mp11::mp_identity<T>, mp11::mp_size_t<N>) const noexcept {
    return launder_cast<const T*>(&buffer_);
  }

  void init_default() noexcept { init_i<default_type>(mp11::mp_size_t<0>{}); }

  template <class T, class I, class... Args>
  void init_i(I, Args&&... args) {
    new (&buffer_) T(std::forward<Args>(args)...);
    index_ = I::value;
  }

  void destroy() noexcept {
    internal_apply([this](auto i) {
      using T = mp11::mp_at_c<variant, i>;
      this->ptr(mp11::mp_identity<T>{}, i)->~T();
    });
  }

  template <class U>
  static constexpr auto find() noexcept {
    using V = std::decay_t<U>;
    return mp11::mp_find<variant, V>{};
  }

  using buffer_t = typename std::aligned_union<0, Ts...>::type;
  buffer_t buffer_;
  unsigned index_;
};

template <class CharT, class Traits, class... Ts>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const variant<Ts...>& x) {
  x.apply([&os](const auto& self) { os << self; });
  return os;
}

// specialization for empty type list, useful for metaprogramming
template <>
class variant<> {};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
