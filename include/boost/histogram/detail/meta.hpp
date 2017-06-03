// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_META_HPP_
#define _BOOST_HISTOGRAM_DETAIL_META_HPP_

#include <boost/fusion/algorithm/iteration/fold.hpp>
#include <boost/fusion/include/fold.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/next.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/unique.hpp>
#include <boost/mpl/vector.hpp>

#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

template <typename T, typename = decltype(std::declval<T &>().size(),
                                          std::declval<T &>().increase(0),
                                          std::declval<T &>().value(0))>
struct is_storage {};

template <typename T,
          typename = decltype(*std::declval<T &>(), ++std::declval<T &>())>
struct is_iterator {};

template <typename T, typename = decltype(std::begin(std::declval<T &>()),
                                          std::end(std::declval<T &>()))>
struct is_sequence {};

template <typename MainVector, typename AuxVector> struct combine {
  using type =
      typename mpl::copy_if<AuxVector,
                            mpl::not_<mpl::contains<MainVector, mpl::_1>>,
                            mpl::back_inserter<MainVector>>::type;
};

struct bool_mask_helper {
  std::vector<bool> &b;
  bool v;
  template <typename N> void operator()(const N &) const { b[N::value] = v; }
};

template <typename Ns> std::vector<bool> bool_mask(std::size_t n, bool v) {
  std::vector<bool> b(n, !v);
  mpl::for_each<Ns>(bool_mask_helper{b, v});
  return b;
}

template <typename Axes, typename Ns> struct axes_assign_subset_helper {
  const Axes &axes_;
  template <typename I, typename R>
  auto operator()(const I &, R &r) const -> typename mpl::next<I>::type {
    using I2 = typename mpl::at_c<Ns, I::value>::type;
    r = fusion::at_c<I2::value>(axes_);
    return {};
  }
};

template <typename Ns, typename Axes1, typename Axes>
void axes_assign_subset(Axes1 &axes1, const Axes &axes) {
  fusion::fold(axes1, mpl::int_<0>(),
               axes_assign_subset_helper<Axes, Ns>{axes});
}

template <typename... Ns> struct unique_sorted {
  using type = typename mpl::unique<
      typename mpl::sort<typename mpl::vector<Ns...>::type>::type>::type;
};

template <typename Axes, typename Numbers> struct axes_select {
  using type = typename mpl::transform<Numbers, mpl::at<Axes, mpl::_1>>::type;
};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
