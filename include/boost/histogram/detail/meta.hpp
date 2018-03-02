// Copyright 2015-2017 Hans Dembinski
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
#include <boost/mpl/int.hpp>
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
                                          std::declval<T &>()[0])>
struct requires_storage {};

template <typename T> struct has_variance_support {
  template <typename U, typename = decltype(std::declval<U&>().value(),
                                            std::declval<U&>().variance())>
  struct SFINAE {};
  template <typename U> static std::true_type Test(SFINAE<U> *);
  template <typename U> static std::false_type Test(...);
  using type = decltype(Test<T>(nullptr));
};

template <typename T>
using has_variance_support_t = typename has_variance_support<T>::type;

template <typename T,
          typename = decltype(*std::declval<T &>(), ++std::declval<T &>())>
struct is_iterator {};

template <typename T, typename = decltype(std::begin(std::declval<T &>()),
                                          std::end(std::declval<T &>()))>
struct is_sequence {};

template <typename MainVector, typename AuxVector> struct union_ :
  mpl::copy_if<AuxVector,
               mpl::not_<mpl::contains<MainVector, mpl::_1>>,
               mpl::back_inserter<MainVector>>
{};

template <typename MainVector, typename AuxVector>
using union_t = typename union_<MainVector, AuxVector>::type;

struct bool_mask_op {
  std::vector<bool> &b;
  bool v;
  template <typename N> void operator()(const N &) const { b[N::value] = v; }
};

template <typename Ns> std::vector<bool> bool_mask(unsigned n, bool v) {
  std::vector<bool> b(n, !v);
  mpl::for_each<Ns>(bool_mask_op{b, v});
  return b;
}

template <typename Axes, typename Ns> struct axes_assign_subset_op {
  const Axes &axes_;
  template <int N, typename R>
  auto operator()(mpl::int_<N>, R &r) const -> mpl::int_<N + 1> {
    using I2 = typename mpl::at_c<Ns, N>::type;
    r = fusion::at_c<I2::value>(axes_);
    return {};
  }
};

template <typename Ns, typename Axes1, typename Axes>
void axes_assign_subset(Axes1 &axes1, const Axes &axes) {
  fusion::fold(axes1, mpl::int_<0>(), axes_assign_subset_op<Axes, Ns>{axes});
}

template <typename Ns>
using unique_sorted_t =
    typename mpl::unique<typename mpl::sort<Ns>::type,
                         std::is_same<mpl::_1, mpl::_2>>::type;

template <typename Axes, typename Numbers>
using axes_select_t =
    typename mpl::transform<Numbers, mpl::at<Axes, mpl::_>>::type;

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
