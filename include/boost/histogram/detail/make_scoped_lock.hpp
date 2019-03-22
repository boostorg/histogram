// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_MAKE_SCOPED_LOCK_HPP
#define BOOST_HISTOGRAM_DETAIL_MAKE_SCOPED_LOCK_HPP

#include <mutex>

namespace boost {
namespace histogram {
namespace detail {

template <class Mutex>
std::lock_guard<Mutex> make_scoped_lock(Mutex& m) {
  return std::lock_guard<Mutex>(m);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
