// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_MPL_HPP_
#define _BOOST_HISTOGRAM_DETAIL_MPL_HPP_

#include <type_traits>
#include <iterator>
#include <limits>
#include <boost/histogram/dynamic_storage.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/logical.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back_inserter.hpp>


namespace boost {
namespace histogram {
namespace detail {

    template <typename T,
              typename = decltype(*std::declval<T&>()),
              typename = decltype(++std::declval<T&>())>
    struct is_iterator {};

    template <typename T,
              typename = decltype(std::begin(std::declval<T&>())),
              typename = decltype(std::end(std::declval<T&>()))>
    struct is_sequence {};

    // prefer dynamic over static storage, choose 
    // static_storage with larger capacity
    template <typename Storage1,
              typename Storage2>
    struct select_storage {
        using type = typename std::conditional<
            (mpl::or_<
                std::is_same<dynamic_storage, Storage1>,
                std::is_same<dynamic_storage, Storage2>
             >::value),
            dynamic_storage,
            typename std::conditional<
                (std::numeric_limits<typename Storage1::value_t>::max() >
                 std::numeric_limits<typename Storage2::value_t>::max()),
                Storage1, Storage2
            >::type
        >::type;
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

}
}
}

#endif
