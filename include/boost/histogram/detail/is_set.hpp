// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_IS_SET_HPP
#define BOOST_HISTOGRAM_DETAIL_IS_SET_HPP

#include <boost/container/flat_set.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp> // for axis::limit

namespace boost {
namespace histogram {
namespace detail {
template <class Iterator>
bool is_set(Iterator begin, Iterator end) {
  using T = iterator_value_type<Iterator>;
  using C = boost::container::static_vector<T, axis::limit>;
  boost::container::flat_set<T, std::less<T>, C> s(begin, end);
  return static_cast<std::size_t>(std::distance(begin, end)) == s.size();
}
} // namespace detail
} // namespace histogram
} // namespace boost

#endif
