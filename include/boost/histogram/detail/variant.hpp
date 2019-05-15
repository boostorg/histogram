// Copyright (c) 2019 Hans Dembinski
// Copyright (c) 2019 Glen Joseph Fernandes (glenjofe@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_VARIANT_HPP
#define BOOST_HISTOGRAM_DETAIL_VARIANT_HPP

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
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
// * No empty state
// * All held types must have copy ctors and copy assignment.
// * All held types must have noexcept move ctors and noexcept move assignment.
//
template <class... Ts>
class variant {
public:
  variant() { create<mp11::mp_first<variant>>(0); }

  variant(variant&& x) noexcept {
    x.internal_visit([this, &x](auto i) {
      using T = mp11::mp_at_c<variant, i>;
      static_assert(std::is_nothrow_move_constructible<T>::value, "");
      this->create<T>(i, std::move(x.template ref<T>()));
    });
  }

  variant& operator=(variant&& x) noexcept {
    x.internal_visit([this, &x](auto i) {
      using T = mp11::mp_at_c<variant, i>;
      this->operator=(std::move(x.template ref<T>()));
    });
    return *this;
  }

  variant(const variant& x) {
    x.internal_visit([this, &x](auto i) {
      using T = mp11::mp_at_c<variant, i>;
      this->create<T>(i, x.template ref<T>());
    });
  }

  variant& operator=(const variant& x) {
    x.visit([this](const auto& x) { this->operator=(x); });
    return *this;
  }

  template <class U, class = std::enable_if_t<(mp11::mp_contains<variant, U>::value &&
                                               !std::is_reference<U>::value)>>
  explicit variant(U&& x) noexcept {
    constexpr auto i = find<U>();
    using T = mp11::mp_at_c<variant, i>;
    static_assert(std::is_nothrow_move_constructible<T>::value, "");
    create<T>(i, std::move(x));
  }

  template <class U, class = std::enable_if_t<(mp11::mp_contains<variant, U>::value &&
                                               !std::is_reference<U>::value)>>
  variant& operator=(U&& x) noexcept {
    constexpr auto i = find<U>();
    using T = mp11::mp_at_c<variant, i>;
    if (i == index_) {
      static_assert(std::is_nothrow_move_assignable<T>::value, "");
      ref<T>() = std::move(x);
    } else {
      destroy();
      static_assert(std::is_nothrow_move_constructible<T>::value, "");
      create<T>(i, std::move(x));
    }
    return *this;
  }

  template <class U, class = std::enable_if_t<mp11::mp_contains<variant, U>::value>>
  explicit variant(const U& x) {
    constexpr auto i = find<U>();
    using T = mp11::mp_at_c<variant, i>;
    create<T>(i, x);
  }

  template <class U, class = std::enable_if_t<mp11::mp_contains<variant, U>::value>>
  variant& operator=(const U& x) {
    auto tp = get_if<U>();
    if (tp) {
      *tp = x;
    } else {
      // Avoid empty state by first calling copy ctor and use move assignment.
      // Copy ctor may throw, leaving variant in current state. If copy ctor succeeds,
      // move assignment is noexcept.
      variant v(x);
      operator=(std::move(v));
    }
    return *this;
  }

  ~variant() { destroy(); }

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
    return i == index_ ? &ref<T>() : nullptr;
  }

  template <class T>
  const T* get_if() const noexcept {
    constexpr auto i = mp11::mp_find<variant, T>{};
    return i == index_ ? &ref<T>() : nullptr;
  }

  template <class Functor>
  decltype(auto) visit(Functor&& functor) const {
    return internal_visit([this, &functor](auto i) -> decltype(auto) {
      using T = mp11::mp_at_c<variant, i>;
      return functor(this->template ref<T>());
    });
  }

  template <class Functor>
  decltype(auto) visit(Functor&& functor) {
    return internal_visit([this, &functor](auto i) -> decltype(auto) {
      using T = mp11::mp_at_c<variant, i>;
      return functor(this->template ref<T>());
    });
  }

private:
  template <class Functor>
  decltype(auto) internal_visit(Functor&& functor) const {
    return mp11::mp_with_index<sizeof...(Ts)>(index_, functor);
  }

  template <class T>
  T& ref() noexcept {
    return launder_cast<T&>(buffer_);
  }

  template <class T>
  const T& ref() const noexcept {
    return launder_cast<const T&>(buffer_);
  }

  template <class T, class... Args>
  void create(unsigned i, Args&&... args) {
    new (&buffer_) T(std::forward<Args>(args)...);
    index_ = i;
  }

  void destroy() {
    internal_visit([this](auto i) {
      using T = mp11::mp_at_c<variant, i>;
      this->template ref<T>().~T();
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
  x.visit([&os](const auto& self) { os << self; });
  return os;
}

// specialization for empty type list, useful for metaprogramming
template <>
class variant<> {};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
