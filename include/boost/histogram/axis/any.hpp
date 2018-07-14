// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_ANY_HPP_
#define _BOOST_HISTOGRAM_AXIS_ANY_HPP_

#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/utility/string_view.hpp>
#include <boost/variant.hpp>
#include <boost/mp11.hpp>
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

struct size_visitor : public static_visitor<int> {
  template <typename A> int operator()(const A &a) const { return a.size(); }
};

struct shape_visitor : public static_visitor<int> {
  template <typename A> int operator()(const A &a) const { return a.shape(); }
};

struct uoflow_visitor : public static_visitor<bool> {
  template <typename A> bool operator()(const A &a) const { return a.uoflow(); }
};

struct get_label_visitor : public static_visitor<string_view> {
  template <typename A>::boost::string_view operator()(const A &a) const {
    return a.label();
  }
};

struct set_label_visitor : public static_visitor<void> {
  const ::boost::string_view label;
  set_label_visitor(const ::boost::string_view x) : label(x) {}
  template <typename A> void operator()(A &a) const { a.label(label); }
};

struct index_visitor : public static_visitor<int> {
  const double x;
  explicit index_visitor(const double arg) : x(arg) {}
  template <typename Axis> int operator()(const Axis &a) const {
    return impl(std::is_convertible<double, typename Axis::value_type>(), a);
  }
  template <typename Axis> int impl(std::true_type, const Axis &a) const {
    return a.index(x);
  }
  template <typename Axis> int impl(std::false_type, const Axis &) const {
    throw std::runtime_error(::boost::histogram::detail::cat(
        "cannot convert double to value_type ",
        boost::typeindex::type_id<typename Axis::value_type>().pretty_name(),
        " of ", boost::typeindex::type_id<Axis>().pretty_name()));
  }
};

struct lower_visitor : public static_visitor<double> {
  int idx;
  lower_visitor(int i) : idx(i) {}
  template <typename Axis> double operator()(const Axis &a) const {
    return impl(
        std::integral_constant<
            bool,
            (std::is_convertible<typename Axis::value_type, double>::value &&
             std::is_same<typename Axis::bin_type,
                          interval_view<Axis>>::value)>(),
        a);
  }
  template <typename Axis> double impl(std::true_type, const Axis &a) const {
    return a.lower(idx);
  }
  template <typename Axis> double impl(std::false_type, const Axis &) const {
    throw std::runtime_error(::boost::histogram::detail::cat(
        "cannot use ", boost::typeindex::type_id<Axis>().pretty_name(),
        " with generic boost::histogram::axis::any interface, use"
        " boost::histogram::axis::cast to access underlying axis type"));
  }
};

struct bicmp_visitor : public static_visitor<bool> {
  template <typename T, typename U> bool operator()(const T&, const U&) const {
    return false;
  }

  template <typename T> bool operator()(const T& t, const T& u) const {
    return t == u;
  }  
};

template <typename T>
struct cmp_visitor : public static_visitor<bool> {
  const T& t;
  cmp_visitor(const T& tt) : t(tt) {}
  template <typename U> bool operator()(const U& u) const {
    return impl(mp11::mp_same<T, U>(), u);
  }

  template <typename U> bool impl(mp11::mp_true, const U& u) const {
    return t == u;
  }

  template <typename U> bool impl(mp11::mp_false, const U&) const {
    return false;
  }
};

template <typename T>
struct assign_visitor : public static_visitor<void> {
  T& t;
  assign_visitor(T& tt) : t(tt) {}
  template <typename U> void operator()(const U& u) const {
    impl(mp11::mp_contains<typename T::types, U>(), u);
  }

  template <typename U> void impl(mp11::mp_true, const U& u) const {
    t = u;
  }

  template <typename U> void impl(mp11::mp_false, const U&) const {
    throw std::invalid_argument(::boost::histogram::detail::cat(
        "argument ",
        boost::typeindex::type_id<U>().pretty_name(),
        " is not a bounded type of ",
        boost::typeindex::type_id<T>().pretty_name()));
  }
};

} // namespace detail

/// Polymorphic axis type
template <typename... Ts> class any : public ::boost::variant<Ts...> {
  using base_type = ::boost::variant<Ts...>;

public:
  using types = mp11::mp_list<Ts...>;
  using value_type = double;
  using bin_type = interval_view<any>;
  using const_iterator = iterator_over<any>;
  using const_reverse_iterator = reverse_iterator_over<any>;

private:
  template <typename T> using requires_bounded_type = mp11::mp_if<
      mp11::mp_contains<types, ::boost::histogram::detail::rm_cv_ref<T>
    >, void>;

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
  any(const any<Us...> & u) {
    ::boost::apply_visitor(detail::assign_visitor<any>(*this), u);
  }

  template <typename... Us>
  any &operator=(const any<Us...> & u) {
    ::boost::apply_visitor(detail::assign_visitor<any>(*this), u);
    return *this;
  }

  int size() const { return ::boost::apply_visitor(detail::size_visitor(), *this); }

  int shape() const { return ::boost::apply_visitor(detail::shape_visitor(), *this); }

  bool uoflow() const { return ::boost::apply_visitor(detail::uoflow_visitor(), *this); }

  // note: this only works for axes with compatible value type
  int index(const value_type x) const {
    return ::boost::apply_visitor(detail::index_visitor(x), *this);
  }

  string_view label() const {
    return ::boost::apply_visitor(detail::get_label_visitor(), *this);
  }

  void label(const string_view x) {
    return ::boost::apply_visitor(detail::set_label_visitor(x), *this);
  }

  // this only works for axes with compatible bin type
  // and will throw a runtime_error otherwise
  double lower(int idx) const {
    return ::boost::apply_visitor(detail::lower_visitor(idx), *this);
  }

  bin_type operator[](const int idx) const { return bin_type(idx, *this); }

  bool operator==(const any &rhs) const {
    return base_type::operator==(static_cast<const base_type &>(rhs));
  }

  template <typename... Us>
  bool operator==(const any<Us...>& u) const {
    return ::boost::apply_visitor(detail::bicmp_visitor(), *this, u);
  }

  template <typename T, typename = requires_bounded_type<T>>
  bool operator==(const T & t) const {
    // variant::operator==(T) is implemented, but only to fail, cannot use it
    return ::boost::apply_visitor(detail::cmp_visitor<T>(t), *this);
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
  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

// dynamic casts
template <typename T, typename... Ts>
typename std::add_lvalue_reference<T>::type cast(any<Ts...> &any) {
  return get<T>(any);
}

template <typename T, typename... Ts>
const typename std::add_lvalue_reference<T>::type cast(const any<Ts...> &any) {
  return get<T>(any);
}

template <typename T, typename... Ts>
typename std::add_pointer<T>::type cast(any<Ts...> *any) {
  return get<T>(&any);
}

template <typename T, typename... Ts>
const typename std::add_pointer<T>::type cast(const any<Ts...> *any) {
  return get<T>(&any);
}

// pass-through for generic programming, to keep code workgin when
// you switch from dynamic to static histogram
template <typename, typename U> auto cast(U && u) -> decltype(std::forward<U>(u)) {
  return std::forward<U>(u);
}

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
