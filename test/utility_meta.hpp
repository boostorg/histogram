// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_META_HPP
#define BOOST_HISTOGRAM_TEST_UTILITY_META_HPP

#include <array>
#include <boost/core/is_same.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/mp11/tuple.hpp>
#include <boost/mp11/utility.hpp>
#include <ostream>
#include <vector>

namespace std {
// never add to std, we only do it here to get ADL working :(
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  os << "[ ";
  for (const auto& x : v) os << x << " ";
  os << "]";
  return os;
}

template <class T,
          class = std::enable_if_t<boost::histogram::detail::has_fixed_size<T>::value>>
ostream& operator<<(ostream& os, const T& t) {
  os << "[ ";
  ::boost::mp11::tuple_for_each(t, [&os](const auto& x) { os << x << " "; });
  os << "]";
  return os;
}
} // namespace std

#ifndef BOOST_TEST_TRAIT_SAME
// temporary copy of macro implementation from boost::core develop branch
// so that BOOST_TEST_TRAIT_SAME also works on travis and appveyor with boost-1.69.0
#define BOOST_TEST_TRAIT_SAME(...)                                              \
  (::boost::detail::test_trait_same_impl(#__VA_ARGS__,                          \
                                         ::boost::core::is_same<__VA_ARGS__>(), \
                                         __FILE__, __LINE__, BOOST_CURRENT_FUNCTION))
#endif

#endif
