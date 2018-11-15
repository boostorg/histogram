// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_IS_SET_HPP
#define BOOST_HISTOGRAM_DETAIL_IS_SET_HPP

#include <algorithm>
#include <boost/container/static_vector.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>

namespace boost {
namespace histogram {
namespace detail {
template <class Iterator>
bool is_set(Iterator begin, Iterator end) {
  using T = iterator_value_type<Iterator>;
  boost::container::static_vector<T, axis::limit> v(begin, end);
  std::sort(v.begin(), v.end());
  auto end2 = std::unique(v.begin(), v.end());
  return static_cast<std::size_t>(std::distance(v.begin(), end2)) == v.size();
}
} // namespace detail
} // namespace histogram
} // namespace boost

#endif
