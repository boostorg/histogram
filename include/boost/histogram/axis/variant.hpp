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

namespace boost {
namespace histogram {

namespace detail {

// struct call_visitor : public boost::static_visitor<int> {
//   const double x;
//   explicit call_visitor(const double arg) : x(arg) {}
//   template <typename T>
//   int operator()(const T& t) const {
//     using args = typename axis::traits<T>::call_args;
//     return impl(mp11::mp_bool<
//                   (mp11::mp_size<args>::value == 1 &&
//                    std::is_convertible<double, mp11::mp_first<args>>::value)
//                 >(), t);
//   }
//   template <typename T>
//   int impl(std::true_type, const T& a) const {
//     using U = mp11::mp_first<typename axis::traits<T>::call_args>;
//     return a(static_cast<U>(x));
//   }
//   template <typename T>
//   int impl(std::false_type, const T&) const {
//     using args = typename axis::traits<T>::call_args;
//     throw std::runtime_error(boost::histogram::detail::cat(
//       "cannot convert double to ",
//       boost::core::demangled_name( BOOST_CORE_TYPEID(args) ),
//       " for ",
//       boost::core::demangled_name( BOOST_CORE_TYPEID(T) )
//     ));
//   }
// };

struct size_visitor : public boost::static_visitor<unsigned> {
  template <typename T>
  unsigned operator()(const T& a) const {
    return a.size();
  }
};

struct options_visitor : public boost::static_visitor<axis::option_type> {
  template <typename T>
  axis::option_type operator()(const T& t) const {
    return axis::traits<T>::options(t);
  }
};

struct lower_visitor : public boost::static_visitor<double> {
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
    throw std::runtime_error(boost::histogram::detail::cat(
        "cannot use ", boost::core::demangled_name( BOOST_CORE_TYPEID(T) ),
        " with generic boost::histogram::axis::variant interface, use"
        " static_cast to access the underlying axis type"));
  }
};

struct bicmp_visitor : public boost::static_visitor<bool> {
  template <typename T, typename U>
  bool operator()(const T&, const U&) const {
    return false;
  }

  template <typename T>
  bool operator()(const T& a, const T& b) const {
    return a == b;
  }
};

template <typename T>
struct assign_visitor : public boost::static_visitor<void> {
  T& t;
  assign_visitor(T& tt) : t(tt) {}
  template <typename U>
  void operator()(const U& u) const {
    impl(mp11::mp_contains<typename T::types, U>(), u);
  }

  template <typename U>
  void impl(std::true_type, const U& u) const {
    t = u;
  }

  template <typename U>
  void impl(std::false_type, const U&) const {
    throw std::invalid_argument(boost::histogram::detail::cat(
        "argument ", boost::typeindex::type_id<U>().pretty_name(),
        " is not a bounded type of ", boost::typeindex::type_id<T>().pretty_name()));
  }
};
} // namespace detail

namespace axis {

/// Polymorphic axis type
template <typename... Ts>
class variant
  : public iterator_mixin<variant<Ts...>>
  , private boost::variant<Ts...>
{
  using base_type = boost::variant<Ts...>;
  using bin_type = interval_view<variant>;

  template <typename T>
  using requires_bounded_type =
    mp11::mp_if<mp11::mp_contains<base_type,
                boost::histogram::detail::rm_cv_ref<T>>, void>;

public:
  using types = mp11::mp_list<Ts...>;

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
    boost::apply_visitor(detail::assign_visitor<variant>(*this), u);
  }

  template <typename... Us>
  variant& operator=(const variant<Us...>& u) {
    boost::apply_visitor(detail::assign_visitor<variant>(*this), u);
    return *this;
  }

  unsigned size() const {
    return boost::apply_visitor(detail::size_visitor(), *this);
  }

  option_type options() const {
    return boost::apply_visitor(detail::options_visitor(), *this);
  }

  // // this only works for axes with compatible call signature
  // // and will throw a runtime_error otherwise
  // int operator()(double x) const {
  //   return boost::apply_visitor(detail::call_visitor(x), *this);
  // }

  // this only works for axes with a lower method
  // and will throw a runtime_error otherwise
  double lower(int idx) const {
    return boost::apply_visitor(detail::lower_visitor(idx), *this);
  }

  // this only works for axes with compatible bin type
  // and will throw a runtime_error otherwise
  bin_type operator[](const int idx) const { return bin_type(idx, *this); }

  bool operator==(const variant& rhs) const {
    return base_type::operator==(static_cast<const base_type&>(rhs));
  }

  template <typename... Us>
  bool operator==(const variant<Us...>& u) const {
    return boost::apply_visitor(detail::bicmp_visitor(), *this, u);
  }

  template <typename T, typename = requires_bounded_type<T>>
  bool operator==(const T& t) const {
    // variant::operator==(T) is implemented only to fail, we cannot use it
    auto tp = boost::get<T>(this);
    return tp && *tp == t;
  }

  template <typename T>
  bool operator!=(const T& t) const {
    return !operator==(t);
  }

  template <typename T>
  explicit operator const T&() const {
    return boost::strict_get<T>(*this);
  }

  template <typename T>
  explicit operator T&() {
    return boost::strict_get<T>(*this);
  }

  template <typename Archive>
  void serialize(Archive&, unsigned);
};

// TODO add something like std::visit

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
