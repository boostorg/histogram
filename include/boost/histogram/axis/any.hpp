// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_ANY_HPP_
#define _BOOST_HISTOGRAM_AXIS_ANY_HPP_

#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/mp11.hpp>
#include <boost/utility/string_view.hpp>
#include <boost/variant.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
} // namespace serialization
} // namespace boost

namespace boost {
namespace histogram {
namespace axis {

namespace detail {

// this visitation may be collapsed by the compiler almost into a direct cast,
// works with gcc-8 -O2 -DNDEBUG on Compiler Explorer at least
template <typename T>
struct static_cast_visitor : public boost::static_visitor<T> {
  template <typename A>
  T operator()(A&& a) const {
    return static_cast<T>(std::forward<A>(a));
  }
};

struct index_visitor : public boost::static_visitor<int> {
  const double x;
  explicit index_visitor(const double arg) : x(arg) {}
  template <typename Axis>
  int operator()(const Axis& a) const {
    return impl(std::is_convertible<double, typename Axis::value_type>(), a);
  }
  template <typename Axis>
  int impl(std::true_type, const Axis& a) const {
    return a.index(static_cast<typename Axis::value_type>(x));
  }
  template <typename Axis>
  int impl(std::false_type, const Axis&) const {
    throw std::runtime_error(boost::histogram::detail::cat(
        "cannot convert double to ",
        boost::typeindex::type_id<typename Axis::value_type>().pretty_name(),
        " for ", boost::typeindex::type_id<Axis>().pretty_name()));
  }
};

struct lower_visitor : public boost::static_visitor<double> {
  int idx;
  lower_visitor(int i) : idx(i) {}
  template <typename Axis>
  double operator()(const Axis& a) const {
    return impl(
        std::integral_constant<
            bool,
            (std::is_convertible<typename Axis::value_type, double>::value &&
             std::is_same<typename Axis::bin_type,
                          interval_view<Axis>>::value)>(),
        a);
  }
  template <typename Axis>
  double impl(std::true_type, const Axis& a) const {
    return a.lower(idx);
  }
  template <typename Axis>
  double impl(std::false_type, const Axis&) const {
    throw std::runtime_error(boost::histogram::detail::cat(
        "cannot use ", boost::typeindex::type_id<Axis>().pretty_name(),
        " with generic boost::histogram::axis::any interface, use"
        " a static_cast to access the underlying axis type"));
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
  void impl(mp11::mp_true, const U& u) const {
    t = u;
  }

  template <typename U>
  void impl(mp11::mp_false, const U&) const {
    throw std::invalid_argument(boost::histogram::detail::cat(
        "argument ", boost::typeindex::type_id<U>().pretty_name(),
        " is not a bounded type of ",
        boost::typeindex::type_id<T>().pretty_name()));
  }
};

} // namespace detail

/// Polymorphic axis type
template <typename... Ts>
class any : public boost::variant<Ts...> {
  using base_type = boost::variant<Ts...>;

public:
  using types = mp11::mp_list<Ts...>;
  using value_type = double;
  using bin_type = interval_view<any>;
  using const_iterator = iterator_over<any>;
  using const_reverse_iterator = reverse_iterator_over<any>;

private:
  template <typename T>
  using requires_bounded_type = mp11::mp_if<
      mp11::mp_contains<types, boost::histogram::detail::rm_cv_ref<T>>, void>;

public:
  any() = default;
  any(const any&) = default;
  any& operator=(const any&) = default;
  any(any&&) = default;
  any& operator=(any&&) = default;

  template <typename T, typename = requires_bounded_type<T>>
  any(T&& t) : base_type(std::forward<T>(t)) {}

  template <typename T, typename = requires_bounded_type<T>>
  any& operator=(T&& t) {
    base_type::operator=(std::forward<T>(t));
    return *this;
  }

  template <typename... Us>
  any(const any<Us...>& u) {
    boost::apply_visitor(detail::assign_visitor<any>(*this), u);
  }

  template <typename... Us>
  any& operator=(const any<Us...>& u) {
    boost::apply_visitor(detail::assign_visitor<any>(*this), u);
    return *this;
  }

  int size() const { return static_cast<const base&>(*this).size(); }

  int shape() const { return static_cast<const base&>(*this).shape(); }

  bool uoflow() const { return static_cast<const base&>(*this).uoflow(); }

  // note: this only works for axes with compatible value type
  int index(const value_type x) const {
    return boost::apply_visitor(detail::index_visitor(x), *this);
  }

  string_view label() const {
    return static_cast<const base&>(*this).label();
  }

  void label(const string_view x) { static_cast<base&>(*this).label(x); }

  // this only works for axes with compatible bin type
  // and will throw a runtime_error otherwise
  double lower(int idx) const {
    return boost::apply_visitor(detail::lower_visitor(idx), *this);
  }

  bin_type operator[](const int idx) const { return bin_type(idx, *this); }

  bool operator==(const any& rhs) const {
    return base_type::operator==(static_cast<const base_type&>(rhs));
  }

  template <typename... Us>
  bool operator==(const any<Us...>& u) const {
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

  explicit operator const base&() const {
    return boost::apply_visitor(detail::static_cast_visitor<const base&>(),
                                *this);
  }

  explicit operator base&() {
    return boost::apply_visitor(detail::static_cast_visitor<base&>(), *this);
  }

  template <typename T>
  explicit operator const T&() const {
    return boost::strict_get<T>(*this);
  }

  template <typename T>
  explicit operator T&() {
    return boost::strict_get<T>(*this);
  }

  const_iterator begin() const { return const_iterator(*this, 0); }
  const_iterator end() const { return const_iterator(*this, size()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(*this, size());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(*this, 0);
  }

private:
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive&, unsigned);
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
