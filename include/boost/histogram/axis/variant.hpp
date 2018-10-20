// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_VARIANT_HPP
#define BOOST_HISTOGRAM_AXIS_VARIANT_HPP

#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/mp11.hpp>
#include <boost/variant.hpp>
#include <boost/core/typeinfo.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <ostream>

namespace boost {
namespace histogram {

namespace detail {

struct call_visitor {
  const double x;
  call_visitor(const double arg) : x(arg) {}
  template <typename T>
  int operator()(const T& t) const {
    using args = typename axis::traits<T>::args;
    return impl(mp11::mp_bool<
                  (mp11::mp_size<args>::value == 1 &&
                   std::is_convertible<double, mp11::mp_first<args>>::value)
                >(), t);
  }
  template <typename T>
  int impl(std::true_type, const T& a) const {
    using U = mp11::mp_first<typename axis::traits<T>::args>;
    return a(static_cast<U>(x));
  }
  template <typename T>
  int impl(std::false_type, const T&) const {
    using args = typename axis::traits<T>::args;
    throw std::invalid_argument(detail::cat(
      "cannot convert double to ",
      boost::core::demangled_name( BOOST_CORE_TYPEID(args) ),
      " for ",
      boost::core::demangled_name( BOOST_CORE_TYPEID(T) )
    ));
  }
};

struct size_visitor {
  template <typename T>
  unsigned operator()(const T& a) const {
    return a.size();
  }
};

struct options_visitor {
  template <typename T>
  axis::option_type operator()(const T& t) const {
    return axis::traits<T>::options(t);
  }
};

struct lower_visitor {
  int idx;
  lower_visitor(int i) : idx(i) {}
  template <typename T>
  double operator()(const T& a) const {
    return impl(detail::has_method_lower<T>(), a);
  }
  template <typename T>
  double impl(std::true_type, const T& a) const {
    return a.lower(idx);
  }
  template <typename T>
  double impl(std::false_type, const T&) const {
    throw std::runtime_error(detail::cat(
        boost::core::demangled_name( BOOST_CORE_TYPEID(T) ),
        " has no lower method"));
  }
};

template <typename T>
struct assign_visitor {
  T& t;
  assign_visitor(T& x) : t(x) {}
  template <typename U>
  void operator()(const U& u) const {
    impl(mp11::mp_contains<rm_cvref<T>, U>(), u);
  }

  template <typename U>
  void impl(std::true_type, const U& u) const {
    t = u;
  }

  template <typename U>
  void impl(std::false_type, const U&) const {
    throw std::invalid_argument(detail::cat(
        boost::core::demangled_name( BOOST_CORE_TYPEID(U) ),
        " is not a bounded type of ",
        boost::core::demangled_name( BOOST_CORE_TYPEID(T) )
      ));
  }
};

template <typename T>
struct equal_visitor {
  const T& t;
  equal_visitor(const T& x) : t(x) {}
  template <typename U>
  bool operator()(const U& u) const {
    return t == u;
  }
};

template <typename OStream>
struct ostream_visitor {
  OStream& os;
  ostream_visitor(OStream& o) : os(o) {}
  template <typename T>
  void operator()(const T& t) const {
    impl(is_streamable<T>(), t);
  }
  template <typename T>
  void impl(std::true_type, const T& t) const {
    os << t;
  }
  template <typename T>
  void impl(std::false_type, const T&) const {
    throw std::invalid_argument(detail::cat(
        boost::core::demangled_name( BOOST_CORE_TYPEID(T) ),
        " is not streamable"));
  }
};

template <typename F, typename R>
struct functor_wrapper : public boost::static_visitor<R> {
  F fcn;
  functor_wrapper(F f) : fcn(std::forward<F>(f)) {}

  template <typename T>
  R operator()(T&& t) const {
    return fcn(std::forward<T>(t));
  }
};

} // namespace detail

namespace axis {

/// Polymorphic axis type
template <typename... Ts>
class variant
  : private boost::variant<Ts...>
  , public iterator_mixin<variant<Ts...>>
{
  using base_type = boost::variant<Ts...>;
  using bin_type = interval_view<variant>;

  template <typename T>
  using requires_bounded_type =
    mp11::mp_if<mp11::mp_contains<base_type,
                detail::rm_cvref<T>>, void>;

public:
  variant() = default;

  template <typename T, typename = requires_bounded_type<T>>
  variant(T&& t) : base_type(std::forward<T>(t)) {}

  template <typename T, typename = requires_bounded_type<T>>
  variant& operator=(T&& t) {
    base_type::operator=(std::forward<T>(t));
    return *this;
  }

  template <typename... Us>
  variant(const variant<Us...>& u) {
    visit(detail::assign_visitor<variant>(*this), u);
  }

  template <typename... Us>
  variant& operator=(const variant<Us...>& u) {
    visit(detail::assign_visitor<variant>(*this), u);
    return *this;
  }

  unsigned size() const {
    return visit(detail::size_visitor(), *this);
  }

  option_type options() const {
    return visit(detail::options_visitor(), *this);
  }

  // Only works for axes with compatible call signature
  // and will throw a invalid_argument exception otherwise
  int operator()(double x) const {
    return visit(detail::call_visitor(x), *this);
  }

  // Only works for axes with a lower method
  // and will throw a runtime_error otherwise
  double lower(int idx) const {
    return visit(detail::lower_visitor(idx), *this);
  }

  // this only works for axes with compatible bin type
  // and will throw a runtime_error otherwise
  bin_type operator[](const int idx) const { return bin_type(idx, *this); }

  bool operator==(const variant& rhs) const {
    return base_type::operator==(static_cast<const base_type&>(rhs));
  }

  template <typename... Us>
  bool operator==(const variant<Us...>& u) const {
    return visit(detail::equal_visitor<decltype(u)>(u), *this);
  }

  template <typename T>
  bool operator==(const T& t) const {
    // boost::variant::operator==(T) implemented only to fail, cannot use it
    auto tp = boost::relaxed_get<T>(this);
    return tp && *tp == t;
  }

  template <typename T>
  bool operator!=(const T& t) const {
    return !operator==(t);
  }

  template <typename Archive>
  void serialize(Archive&, unsigned);

  template <typename Functor, typename Variant>
  friend auto visit(Functor&& f, Variant&& v)
    -> detail::visitor_return_type<Functor, Variant>;

  template <typename T, typename... Us>
  friend T& get(variant<Us...>& v);

  template <typename T, typename... Us>
  friend const T& get(const variant<Us...>& v);

  template <typename T, typename... Us>
  friend T&& get(variant<Us...>&& v);

  template <typename T, typename... Us>
  friend T* get(variant<Us...>* v);

  template <typename T, typename... Us>
  friend const T* get(const variant<Us...>* v);
};

template <typename Functor, typename Variant>
auto visit(Functor&& f, Variant&& v)
  -> detail::visitor_return_type<Functor, Variant>
{
  using R = detail::visitor_return_type<Functor, Variant>;
  return boost::apply_visitor(detail::functor_wrapper<Functor, R>(std::forward<Functor>(f)),
                              static_cast<
                                detail::copy_qualifiers<Variant,
                                  typename detail::rm_cvref<Variant>::base_type>
                              >(v));
}

template <typename CharT, typename Traits, typename... Ts>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, const variant<Ts...>& v)
{
  visit(detail::ostream_visitor<decltype(os)>(os), v);
  return os;
}

template <typename T, typename... Us>
T& get(variant<Us...>& v) {
  return boost::relaxed_get<T>(v);
}

template <typename T, typename... Us>
const T& get(const variant<Us...>& v) {
  return boost::relaxed_get<T>(v);
}

template <typename T, typename... Us>
T&& get(variant<Us...>&& v) {
  return boost::relaxed_get<T>(v);
}

template <typename T, typename... Us>
T* get(variant<Us...>* v) {
  return boost::relaxed_get<T>(v);
}

template <typename T, typename... Us>
const T* get(const variant<Us...>* v) {
  return boost::relaxed_get<T>(v);
}
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
