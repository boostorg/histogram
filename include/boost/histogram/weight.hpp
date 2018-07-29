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
struct weight {
  T value;
};

template <typename T>
struct sample {
  T value;
};
} // namespace detail

template <typename T>
detail::weight<T> weight(T&& t) {
  return {t};
}

template <typename T>
detail::sample<T> sample(T&& t) {
  return {t};
}
}
}

#endif
