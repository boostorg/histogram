// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_WEIGHT_HPP_
#define _BOOST_HISTOGRAM_WEIGHT_HPP_

namespace boost {
namespace histogram {
namespace detail {
template <typename T>
struct weight_type {
  T value;
};

template <typename T>
struct sample_type {
  T value;
};
} // namespace detail

template <typename T>
detail::weight_type<T> weight(T&& t) {
  return {t};
}

template <typename T>
detail::sample_type<T> sample(T&& t) {
  return {t};
}
}
}

#endif
