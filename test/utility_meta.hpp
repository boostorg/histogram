// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_META_HPP
#define BOOST_HISTOGRAM_TEST_UTILITY_META_HPP

#include <boost/core/typeinfo.hpp>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/tuple.hpp>
#include <boost/type.hpp>
#include <ostream>
#include <tuple>
#include <vector>

using i0 = boost::mp11::mp_size_t<0>;
using i1 = boost::mp11::mp_size_t<1>;
using i2 = boost::mp11::mp_size_t<2>;
using i3 = boost::mp11::mp_size_t<3>;

namespace std {
// never add to std, we only do it here to get ADL working :(
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  os << "[ ";
  for (const auto& x : v) os << x << " ";
  os << "]";
  return os;
}

template <typename... Ts>
ostream& operator<<(ostream& os, const tuple<Ts...>& t) {
  os << "[ ";
  ::boost::mp11::tuple_for_each(t, [&os](const auto& x) { os << x << " "; });
  os << "]";
  return os;
}
} // namespace std

namespace boost {
namespace histogram {
template <typename T>
std::string type_name() {
  return boost::core::demangled_name(BOOST_CORE_TYPEID(boost::type<T>));
}

template <typename T>
std::string type_name(T) {
  return boost::core::demangled_name(BOOST_CORE_TYPEID(boost::type<T>));
}
} // namespace histogram
} // namespace boost

#endif
