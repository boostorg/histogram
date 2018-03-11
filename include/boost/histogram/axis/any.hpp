// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_ANY_HPP_
#define _BOOST_HISTOGRAM_AXIS_ANY_HPP_

#include <boost/histogram/axis/bin_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/cat.hpp>
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

struct index : public static_visitor<int> {
  const double x;
  explicit index(const double arg) : x(arg) {}
  template <typename Axis> int operator()(const Axis &a) const {
    return impl(std::is_convertible<double, typename Axis::value_type>(), a);
  }
  template <typename Axis> int impl(std::true_type, const Axis &a) const {
    return a.index(x);
  }
  template <typename Axis> int impl(std::false_type, const Axis &) const {
    throw std::runtime_error(::boost::histogram::detail::cat(
        "cannot convert value_type ",
        boost::typeindex::type_id<typename Axis::value_type>().pretty_name(),
        " of ",
        boost::typeindex::type_id<Axis>().pretty_name(),
        " to double")
      );
  }
};

struct lower : public static_visitor<double> {
  int idx;
  lower(int i) : idx(i) {}
  template <typename Axis> double operator()(const Axis &a) const {
    return impl(std::integral_constant<bool,
        (std::is_convertible<typename Axis::value_type, double>::value &&
         std::is_same<typename Axis::bin_type, interval_view<Axis>>::value)
        >(), a);
  }
  template <typename Axis> double impl(std::true_type, const Axis &a) const {
    return a.lower(idx) ;
  }
  template <typename Axis> double impl(std::false_type, const Axis &) const {
    throw std::runtime_error(::boost::histogram::detail::cat(
        "cannot use ",
        boost::typeindex::type_id<Axis>().pretty_name(),
        " with generic boost::histogram::axis::any interface, use"
        " boost::histogram::axis::cast to access underlying axis type")
      );
  }
};
} // namespace detail

/// Polymorphic axis type
template <typename Axes> class any : public make_variant_over<Axes>::type {
  using base_type = typename make_variant_over<Axes>::type;

public:
  using types = typename base_type::types;
  using value_type = double;
  using bin_type = interval_view<any>;
  using const_iterator = iterator_over<any>;
  using const_reverse_iterator = reverse_iterator_over<any>;

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
    base_type::operator=(t);
    return *this;
  }

  template <typename T, typename = typename std::enable_if<
                            mpl::contains<types, T>::value>::type>
  any &operator=(T &&t) {
    base_type::operator=(std::move(t));
    return *this;
  }

  int size() const { return apply_visitor(detail::size(), *this); }

  int shape() const { return apply_visitor(detail::shape(), *this); }

  bool uoflow() const { return apply_visitor(detail::uoflow(), *this); }

  // note: this only works for axes with compatible value type
  int index(const value_type x) const {
    return apply_visitor(detail::index(x), *this);
  }

  string_view label() const {
    return apply_visitor(detail::get_label(), *this);
  }

  void label(const string_view x) {
    return apply_visitor(detail::set_label(x), *this);
  }

  // this only works for axes with compatible bin type
  // and will throw a runtime_error otherwise
  double lower(int idx) const {
    return apply_visitor(detail::lower(idx), *this);
  }

  bin_type operator[](const int idx) const {
    return bin_type(idx, *this);
  }

  bool operator==(const any &rhs) const {
    return base_type::operator==(static_cast<const base_type &>(rhs));
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
