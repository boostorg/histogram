// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_VARIANT_HPP
#define BOOST_HISTOGRAM_AXIS_VARIANT_HPP

#include <boost/core/typeinfo.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/axis/polymorphic_bin.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11/bind.hpp>
#include <boost/mp11/function.hpp>
#include <boost/mp11/list.hpp>
#include <boost/throw_exception.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/get.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/variant.hpp>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

namespace detail {
template <class F, class R>
struct functor_wrapper : public boost::static_visitor<R> {
  F fcn;
  functor_wrapper(F f) : fcn(f) {}
  template <class T>
  R operator()(T&& t) const {
    return fcn(std::forward<T>(t));
  }
};
} // namespace detail

namespace axis {

template <class T, class... Us>
T* get_if(variant<Us...>* v);

template <class T, class... Us>
const T* get_if(const variant<Us...>* v);

/// Polymorphic axis type
template <class... Ts>
class variant : private boost::variant<Ts...>, public iterator_mixin<variant<Ts...>> {
  using base_type = boost::variant<Ts...>;
  using naked_types = mp11::mp_transform<detail::naked, base_type>;

  template <class T>
  using is_bounded_type = mp11::mp_contains<naked_types, detail::naked<T>>;

  template <typename T>
  using requires_bounded_type = std::enable_if_t<is_bounded_type<T>::value>;

public:
  // maybe metadata_type or const metadata_type, if bounded type is const
  using metadata_type = std::remove_reference_t<decltype(
      traits::metadata(std::declval<mp11::mp_first<base_type>>()))>;

  // cannot import ctors with using directive, it breaks gcc and msvc
  variant() = default;
  variant(const variant&) = default;
  variant& operator=(const variant&) = default;
  variant(variant&&) = default;
  variant& operator=(variant&&) = default;

  template <typename T, typename = requires_bounded_type<T>>
  variant(T&& t) : base_type(std::forward<T>(t)) {}

  template <typename T, typename = requires_bounded_type<T>>
  variant& operator=(T&& t) {
    base_type::operator=(std::forward<T>(t));
    return *this;
  }

  template <class... Us>
  variant(const variant<Us...>& u) {
    this->operator=(u);
  }

  template <class... Us>
  variant& operator=(const variant<Us...>& u) {
    visit(
        [this](const auto& u) {
          using U = detail::naked<decltype(u)>;
          detail::static_if<is_bounded_type<U>>(
              [this](const auto& u) { this->operator=(u); },
              [](const auto&) {
                BOOST_THROW_EXCEPTION(std::runtime_error(detail::cat(
                    boost::core::demangled_name(BOOST_CORE_TYPEID(U)),
                    " is not convertible to a bounded type of ",
                    boost::core::demangled_name(BOOST_CORE_TYPEID(variant)))));
              },
              u);
        },
        u);
    return *this;
  }

  int size() const {
    return visit([](const auto& x) { return x.size(); }, *this);
  }

  option options() const {
    return visit([](const auto& x) { return axis::traits::options(x); }, *this);
  }

  const metadata_type& metadata() const {
    return visit(
        [](const auto& a) -> const metadata_type& {
          using M = decltype(traits::metadata(a));
          return detail::static_if<std::is_same<M, const metadata_type&>>(
              [](const auto& a) -> const metadata_type& { return traits::metadata(a); },
              [](const auto&) -> const metadata_type& {
                BOOST_THROW_EXCEPTION(std::runtime_error(detail::cat(
                    "cannot return metadata of type ",
                    boost::core::demangled_name(BOOST_CORE_TYPEID(M)),
                    " through axis::variant interface which uses type ",
                    boost::core::demangled_name(BOOST_CORE_TYPEID(metadata_type)),
                    "; use boost::histogram::axis::get to obtain a reference "
                    "of this axis type")));
              },
              a);
        },
        *this);
  }

  metadata_type& metadata() {
    return visit(
        [](auto& a) -> metadata_type& {
          using M = decltype(traits::metadata(a));
          return detail::static_if<std::is_same<M, metadata_type&>>(
              [](auto& a) -> metadata_type& { return traits::metadata(a); },
              [](auto&) -> metadata_type& {
                BOOST_THROW_EXCEPTION(std::runtime_error(detail::cat(
                    "cannot return metadata of type ",
                    boost::core::demangled_name(BOOST_CORE_TYPEID(M)),
                    " through axis::variant interface which uses type ",
                    boost::core::demangled_name(BOOST_CORE_TYPEID(metadata_type)),
                    "; use boost::histogram::axis::get to obtain a reference "
                    "of this axis type")));
              },
              a);
        },
        *this);
  }

  // Throws invalid_argument exception if axis has incompatible call signature
  template <class U>
  int operator()(const U& u) const {
    return visit([&u](const auto& a) { return traits::index(a, u); }, *this);
  }

  // Throws invalid_argument exception if axis has incompatible call signature
  template <class U>
  std::pair<int, int> update(const U& u) {
    return visit([&u](auto& a) { return traits::update(a, u); }, *this);
  }

  // Only works for axes with value method that returns something convertible to
  // double and will throw a runtime_error otherwise, see axis::traits::value
  double value(double idx) const {
    return visit([idx](const auto& a) { return traits::value_as<double>(a, idx); },
                 *this);
  }

  auto operator[](const int idx) const {
    return visit(
        [idx](const auto& a) {
          return detail::value_method_switch_with_return_type<double,
                                                              polymorphic_bin<double>>(
              [idx](const auto& a) { // axis is discrete
                const auto x = a.value(idx);
                return polymorphic_bin<double>(x, x);
              },
              [idx](const auto& a) { // axis is continuous
                return polymorphic_bin<double>(a.value(idx), a.value(idx + 1));
              },
              a);
        },
        *this);
  }

  template <class... Us>
  bool operator==(const variant<Us...>& u) const {
    return visit([&u](const auto& x) { return u == x; }, *this);
  }

  template <class T>
  bool operator==(const T& t) const {
    // boost::variant::operator==(T) implemented only to fail, cannot use it
    auto tp = get_if<T>(this);
    return tp && detail::relaxed_equal(*tp, t);
  }

  template <class T>
  bool operator!=(const T& t) const {
    return !operator==(t);
  }

  template <class Archive>
  void serialize(Archive& ar, unsigned);

  template <class Functor, class Variant>
  friend auto visit(Functor&&, Variant &&)
      -> detail::visitor_return_type<Functor, Variant>;

  template <class T, class... Us>
  friend T& get(variant<Us...>& v);

  template <class T, class... Us>
  friend const T& get(const variant<Us...>& v);

  template <class T, class... Us>
  friend T&& get(variant<Us...>&& v);

  template <class T, class... Us>
  friend T* get_if(variant<Us...>* v);

  template <class T, class... Us>
  friend const T* get_if(const variant<Us...>* v);
};

template <class Functor, class Variant>
auto visit(Functor&& f, Variant&& v) -> detail::visitor_return_type<Functor, Variant> {
  using R = detail::visitor_return_type<Functor, Variant>;
  using B = detail::copy_qualifiers<Variant, typename detail::naked<Variant>::base_type>;
  return boost::apply_visitor(detail::functor_wrapper<Functor, R>(f), static_cast<B>(v));
}

template <class T, class... Us>
T& get(variant<Us...>& v) {
  using B = typename variant<Us...>::base_type;
  return boost::get<T>(static_cast<B&>(v));
}

template <class T, class... Us>
T&& get(variant<Us...>&& v) {
  using B = typename variant<Us...>::base_type;
  return boost::get<T>(static_cast<B&&>(v));
}

template <class T, class... Us>
const T& get(const variant<Us...>& v) {
  using B = typename variant<Us...>::base_type;
  return boost::get<T>(static_cast<const B&>(v));
}

template <class T, class... Us>
T* get_if(variant<Us...>* v) {
  using B = typename variant<Us...>::base_type;
  return boost::relaxed_get<T>(static_cast<B*>(v));
}

template <class T, class... Us>
const T* get_if(const variant<Us...>* v) {
  using B = typename variant<Us...>::base_type;
  return boost::relaxed_get<T>(static_cast<const B*>(v));
}

// pass-through version for generic programming, if U is axis instead of variant
template <class T, class U>
decltype(auto) get(U&& u) {
  return static_cast<detail::copy_qualifiers<U, T>>(u);
}

// pass-through version for generic programming, if U is axis instead of variant
template <class T, class U>
T* get_if(U* u) {
  return std::is_same<T, detail::naked<U>>::value ? reinterpret_cast<T*>(u) : nullptr;
}

// pass-through version for generic programming, if U is axis instead of variant
template <class T, class U>
const T* get_if(const U* u) {
  return std::is_same<T, detail::naked<U>>::value ? reinterpret_cast<const T*>(u)
                                                  : nullptr;
}
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
