// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Uses code segments from based on boost/iterator/iterator_adaptor.hpp
// and boost/iterator/iterator_fascade.hpp

#ifndef BOOST_HISTOGRAM_DETAIL_ITERATOR_ADAPTOR_HPP
#define BOOST_HISTOGRAM_DETAIL_ITERATOR_ADAPTOR_HPP

#include <boost/histogram/detail/meta.hpp>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {
namespace detail {

// operator->() needs special support for input iterators to strictly meet the
// standard's requirements. If *i is not a reference type, we must still
// produce an lvalue to which a pointer can be formed.  We do that by
// returning a proxy object containing an instance of the reference object.
template <class Reference>
struct operator_arrow_dispatch_t // proxy references
{
  struct proxy {
    explicit proxy(Reference const& x) : m_ref(x) {}
    Reference* operator->() { return std::addressof(m_ref); }
    Reference m_ref;
  };

  using result_type = proxy;
  static result_type apply(Reference const& x) { return proxy(x); }
};

template <class T>
struct operator_arrow_dispatch_t<T&> // "real" references
{
  using result_type = T*;
  static result_type apply(T& x) { return std::addressof(x); }
};

// only for random access Base
template <class Derived, class Base, class Reference = std::remove_pointer_t<Base>&,
          class Value = remove_cvref_t<Reference>>
class iterator_adaptor {
  using operator_arrow_dispatch = operator_arrow_dispatch_t<Reference>;

public:
  using base_type = Base;

  using reference = Reference;
  using value_type = std::remove_const_t<Value>;
  using pointer = typename operator_arrow_dispatch::result_type;
  using difference_type = decltype(std::declval<Base>() - std::declval<Base>());
  using iterator_category = std::random_access_iterator_tag;

  iterator_adaptor() = default;

  explicit iterator_adaptor(base_type const& iter) : iter_(iter) {}

  pointer operator->() const {
    return operator_arrow_dispatch::apply(this->derived().operator*());
  }
  reference operator[](difference_type n) const { return (*this + n).operator*(); }

  Derived& operator++() {
    ++iter_;
    return this->derived();
  }

  Derived& operator--() {
    --iter_;
    return this->derived();
  }

  Derived operator++(int) {
    Derived tmp(this->derived());
    ++iter_;
    return tmp;
  }

  Derived operator--(int) {
    Derived tmp(this->derived());
    --iter_;
    return tmp;
  }

  Derived& operator+=(difference_type n) {
    iter_ += n;
    return this->derived();
  }

  Derived& operator-=(difference_type n) {
    iter_ -= n;
    return this->derived();
  }

  Derived operator+(difference_type n) const {
    Derived tmp(this->derived());
    tmp += n;
    return tmp;
  }

  Derived operator-(difference_type n) const { return operator+(-n); }

  template <class... Ts>
  difference_type operator-(const iterator_adaptor<Ts...>& x) const noexcept {
    return iter_ - x.iter_;
  }

  template <class... Ts>
  bool operator==(const iterator_adaptor<Ts...>& x) const noexcept {
    return iter_ == x.iter_;
  }
  template <class... Ts>
  bool operator!=(const iterator_adaptor<Ts...>& x) const noexcept {
    return !this->derived().operator==(x); // equal operator may be overridden in derived
  }
  template <class... Ts>
  bool operator<(const iterator_adaptor<Ts...>& x) const noexcept {
    return iter_ < x.iter_;
  }
  template <class... Ts>
  bool operator>(const iterator_adaptor<Ts...>& x) const noexcept {
    return iter_ > x.iter_;
  }
  template <class... Ts>
  bool operator<=(const iterator_adaptor<Ts...>& x) const noexcept {
    return iter_ <= x.iter_;
  }
  template <class... Ts>
  bool operator>=(const iterator_adaptor<Ts...>& x) const noexcept {
    return iter_ >= x.iter_;
  }

  friend Derived operator+(difference_type n, const Derived& x) { return x + n; }

  Base const& base() const { return iter_; }

protected:
  // for convenience in derived classes
  using iterator_adaptor_ = iterator_adaptor;

private:
  Derived& derived() { return *static_cast<Derived*>(this); }
  const Derived& derived() const { return *static_cast<Derived const*>(this); }

  Base iter_;

  template <class, class, class, class>
  friend class iterator_adaptor;
};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
