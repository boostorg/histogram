// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_META_HPP_
#define _BOOST_HISTOGRAM_DETAIL_META_HPP_

#include <type_traits>
#include <iterator>
#include <limits>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/remove_if.hpp>
#include <boost/mpl/logical.hpp>

namespace boost {
namespace histogram {
namespace detail {

template <typename T,
          typename = decltype(std::declval<T&>().size(),
                              std::declval<T&>().increase(0),
                              std::declval<T&>().value(0),
                              std::declval<T&>().variance(0))>
struct is_storage {};

template <typename T,
          typename = decltype(*std::declval<T&>(),
                              ++std::declval<T&>())>
struct is_iterator {};

template <typename T,
          typename = decltype(std::begin(std::declval<T&>()),
                              std::end(std::declval<T&>()))>
struct is_sequence {};

template <typename T,
          typename = typename std::enable_if<
            (std::is_integral<T>::value &&
             (sizeof(T) & (sizeof(T) - 1)) == 0 && // size in 1,2,4,8
             sizeof(T) <= sizeof(uint64_t))
          >::type>
struct is_standard_integral {};

template <typename T>
struct has_weight_support
{
  template <typename>
  static std::false_type test(...);

  template <typename C>
  static decltype((std::declval<C&>().increase(0, 0.0)), std::true_type{}) test(int);

  static bool const value = decltype(test<T>(0))::value;
};

template <typename S1,
          typename S2>
struct intersection {
    using type = typename std::conditional<
        mpl::equal<S1, S2>::value,
        S1,
        typename mpl::remove_if<
            S1,
            mpl::not_<mpl::contains<S2, mpl::_>>
        >::type
    >::type;
};

// // prefer dynamic over static storage, choose
// // static_storage with larger capacity
// template <typename Storage1,
//           typename Storage2>
// struct select_storage {
//     using type = typename std::conditional<
//         (is_dynamic_storage<Storage1>::value ||
//          is_dynamic_storage<Storage2>::value),
//         typename std::conditional<
//             is_dynamic_storage<Storage1>::value,
//             Storage1, Storage2
//         >::type,
//         typename std::conditional<
//             (std::numeric_limits<typename Storage1::value_t>::max() >
//              std::numeric_limits<typename Storage2::value_t>::max()),
//             Storage1, Storage2
//         >::type
//     >::type;
// };

}
}
}

#endif
