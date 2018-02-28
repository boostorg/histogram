// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_ANY_HPP_
#define _BOOST_HISTOGRAM_AXIS_ANY_HPP_

#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/interval.hpp>
#include <boost/mpl/contains.hpp>
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
struct size : public static_visitor<int> {
  template <typename A> int operator()(const A &a) const { return a.size(); }
};

struct shape : public static_visitor<int> {
  template <typename A> int operator()(const A &a) const { return a.shape(); }
};

struct uoflow : public static_visitor<bool> {
  template <typename A> bool operator()(const A &a) const { return a.uoflow(); }
};

struct get_label : public static_visitor<string_view> {
  template <typename A>::boost::string_view operator()(const A &a) const {
    return a.label();
  }
};

struct set_label : public static_visitor<void> {
  const ::boost::string_view label;
  set_label(const ::boost::string_view x) : label(x) {}
  template <typename A> void operator()(A &a) const { a.label(label); }
};

template <typename T> struct index : public static_visitor<int> {
  const T &t;
  explicit index(const T &arg) : t(arg) {}
  template <typename Axis> int operator()(const Axis &a) const {
    return impl(std::is_convertible<T, typename Axis::value_type>(), a);
  }
  template <typename Axis> int impl(std::true_type, const Axis &a) const {
    return a.index(t);
  }
  template <typename Axis> int impl(std::false_type, const Axis &) const {
    throw std::runtime_error(::boost::histogram::detail::cat(
        "fill argument not convertible to axis value type: ",
        boost::typeindex::type_id<Axis>().pretty_name(), ", ",
        boost::typeindex::type_id<T>().pretty_name()));
  }
};

struct bin : public static_visitor<axis::interval<double>> {
  using double_interval = axis::interval<double>;
  const int i;
  bin(const int v) : i(v) {}
  template <typename A> double_interval operator()(const A &a) const {
    return impl(is_convertible<typename A::bin_type, double_interval>(),
                std::forward<typename A::bin_type>(a[i]));
  }
  template <typename B> double_interval impl(true_type, B &&b) const {
    return b;
  }
  template <typename B> double_interval impl(false_type, B &&) const {
    throw std::runtime_error("cannot convert bin_type to interval<double>");
  }
};
} // namespace detail

/// Polymorphic axis type
template <typename Axes> class any : public make_variant_over<Axes>::type {
  using base_type = typename make_variant_over<Axes>::type;

public:
  using types = typename base_type::types;
  using value_type = double;
  using bin_type = interval<double>;
  using const_iterator = iterator_over<any>;

  any() = default;
  any(const any &t) = default;
  any(any &&t) = default;
  any &operator=(const any &t) = default;
  any &operator=(any &&t) = default;

  template <typename T, typename = typename std::enable_if<
                            mpl::contains<types, T>::value>::type>
  any(const T &t) : base_type(t) {}

  template <typename T, typename = typename std::enable_if<
                            mpl::contains<types, T>::value>::type>
  any &operator=(const T &t) {
    // ugly workaround for compiler bug
    return reinterpret_cast<any &>(base_type::operator=(t));
  }

  template <typename T, typename = typename std::enable_if<
                            mpl::contains<types, T>::value>::type>
  any &operator=(T &&t) {
    // ugly workaround for compiler bug
    return reinterpret_cast<any &>(base_type::operator=(std::move(t)));
  }

  int size() const { return apply_visitor(detail::size(), *this); }

  int shape() const { return apply_visitor(detail::shape(), *this); }

  bool uoflow() const { return apply_visitor(detail::uoflow(), *this); }

  // note: this only works for axes with compatible value type
  int index(const value_type x) const {
    return apply_visitor(detail::index<value_type>(x), *this);
  }

  string_view label() const {
    return apply_visitor(detail::get_label(), *this);
  }

  void label(const string_view x) {
    return apply_visitor(detail::set_label(x), *this);
  }

  // this only works for axes with compatible bin type
  // and will raise an error otherwise
  bin_type operator[](const int i) const {
    return apply_visitor(detail::bin(i), *this);
  }

  bool operator==(const any &rhs) const {
    return base_type::operator==(static_cast<const base_type &>(rhs));
  }

  const_iterator begin() const { return const_iterator(*this, 0); }

  const_iterator end() const { return const_iterator(*this, size()); }

private:
  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

// dynamic casts
template <typename T, typename Axes>
typename std::add_lvalue_reference<T>::type cast(any<Axes> &any) {
  return get<T>(any);
}

template <typename T, typename Axes>
const typename std::add_lvalue_reference<T>::type cast(const any<Axes> &any) {
  return get<T>(any);
}

template <typename T, typename Axes>
typename std::add_pointer<T>::type cast(any<Axes> *any) {
  return get<T>(&any);
}

template <typename T, typename Axes>
const typename std::add_pointer<T>::type cast(const any<Axes> *any) {
  return get<T>(&any);
}

// pass-through versions for generic programming, i.e. when you switch to static
// histogram
template <typename T> typename std::add_lvalue_reference<T>::type cast(T &t) {
  return t;
}

template <typename T>
const typename std::add_lvalue_reference<T>::type cast(const T &t) {
  return t;
}

template <typename T> typename std::add_pointer<T>::type cast(T *t) {
  return t;
}

template <typename T>
const typename std::add_pointer<T>::type cast(const T *t) {
  return t;
}

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
