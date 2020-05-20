// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_INDEX_TRANSLATOR_HPP
#define BOOST_HISTOGRAM_DETAIL_INDEX_TRANSLATOR_HPP

#include <boost/assert.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/relaxed_equal.hpp>
#include <boost/histogram/detail/relaxed_tuple_size.hpp>
#include <boost/histogram/multi_index.hpp>
#include <boost/mp11/integer_sequence.hpp>
#include <initializer_list>
#include <tuple>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

template <class A>
struct index_translator {
  const A& dst;
  const A& src;
  using index_type = axis::index_type;
  using multi_index_type = multi_index<relaxed_tuple_size_t<A>::value>;

  template <class T>
  index_type translate(const T& a, const T& b, const index_type& i) const noexcept {
    return axis::traits::index(a, axis::traits::value(b, i + 0.5));
  }

  template <class T, class It, std::size_t... Is>
  void impl(const T& a, const T& b, It i, index_type* j,
            mp11::index_sequence<Is...>) const noexcept {
    // operator folding emulation
    ignore_unused(std::initializer_list<index_type>{
        (*j++ = translate(std::get<Is>(a), std::get<Is>(b), *i++))...});
  }

  template <class... Ts, class It>
  void impl(const std::tuple<Ts...>& a, const std::tuple<Ts...>& b, It i,
            index_type* j) const noexcept {
    impl(a, b, i, j, mp11::make_index_sequence<sizeof...(Ts)>{});
  }

  template <class T, class It>
  void impl(const T& a, const T& b, It i, index_type* j) const noexcept {
    for (unsigned k = 0; k < a.size(); ++k, ++i, ++j) {
      const auto& bk = b[k];
      axis::visit(
          [&](const auto& ak) {
            using U = std::decay_t<decltype(ak)>;
            *j = this->translate(ak, axis::get<U>(bk), *i);
          },
          a[k]);
    }
  }

  template <class Indices>
  auto operator()(const Indices& seq) const noexcept {
    auto mi = multi_index_type::create(seq.size());
    impl(dst, src, seq.begin(), mi.begin());
    return mi;
  }
};

template <class Axes>
auto make_index_translator(const Axes& dst, const Axes& src) noexcept {
  return index_translator<Axes>{dst, src};
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
