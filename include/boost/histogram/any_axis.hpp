// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_ANY_AXIS_HPP_
#define _BOOST_HISTOGRAM_ANY_AXIS_HPP_

#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/variant.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/utility/string_view.hpp>
#include <type_traits>

namespace boost {
namespace histogram {

/// Polymorphic axis type
template <typename Axes>
class any_axis : public make_variant_over<Axes>::type {
public:
  using base_type = typename make_variant_over<Axes>::type;
  any_axis() = default;
  any_axis(const any_axis& t) = default;
  any_axis(any_axis&& t) = default;
  any_axis& operator=(const any_axis& t) = default;
  any_axis& operator=(any_axis&& t) = default;

  template <typename T, typename = typename std::enable_if<
    mpl::contains<Axes, T>::value
  >::type>
  any_axis(const T& t) : base_type(t) {}

  template <typename T, typename = typename std::enable_if<
    mpl::contains<Axes, T>::value
  >::type>
  any_axis& operator=(const T& t) {
    // ugly workaround for compiler bug
    return reinterpret_cast<any_axis&>(base_type::operator=(t));
  }

  template <typename T, typename = typename std::enable_if<
    mpl::contains<Axes, T>::value
  >::type>
  any_axis& operator=(T&& t) {
    // ugly workaround for compiler bug
    return reinterpret_cast<any_axis&>(base_type::operator=(std::move(t)));
  }

  int size() const {
    return apply_visitor(detail::size(), *this);
  }

  int shape() const {
    return apply_visitor(detail::shape(), *this);
  }

  bool uoflow() const {
    return apply_visitor(detail::uoflow(), *this);
  }

  // note: this only works for axes with compatible value type
  int index(const double x) const {
    return apply_visitor(detail::index<double>(x), *this);
  }

  ::boost::string_view label() const {
    return apply_visitor(detail::get_label(), *this);
  }

  void label(const ::boost::string_view x) {
    return apply_visitor(detail::set_label(x), *this);
  }

  // note: this only works for axes with compatible bin type
  axis::interval<double> operator[](const int i) const {
    return apply_visitor(detail::bin(i), *this);
  }

  bool operator==(const any_axis& rhs) const {
    return base_type::operator==(static_cast<const base_type&>(rhs));
  }
};

// dynamic casts
template <typename T, typename Axes>
typename std::add_lvalue_reference<T>::type axis_cast(any_axis<Axes>& any) { return ::boost::get<T>(any); }

template <typename T, typename Axes>
const typename std::add_lvalue_reference<T>::type axis_cast(const any_axis<Axes>& any) { return ::boost::get<T>(any); }

template <typename T, typename Axes>
typename std::add_pointer<T>::type axis_cast(any_axis<Axes>* any) { return ::boost::get<T>(&any); }

template <typename T, typename Axes>
const typename std::add_pointer<T>::type axis_cast(const any_axis<Axes>* any) { return ::boost::get<T>(&any); }

// pass-through versions for generic programming, i.e. when you switch to static histogram
template <typename T>
typename std::add_lvalue_reference<T>::type axis_cast(T& t) { return t; }

template <typename T>
const typename std::add_lvalue_reference<T>::type axis_cast(const T& t) { return t; }

template <typename T>
typename std::add_pointer<T>::type axis_cast(T* t) { return t; }

template <typename T>
const typename std::add_pointer<T>::type axis_cast(const T* t) { return t; }

}
}

#endif
