// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_PYTHON_UTILITY_HPP_
#define _BOOST_HISTOGRAM_PYTHON_UTILITY_HPP_

#include <boost/python/str.hpp>
#include <type_traits>
#include <stdexcept>

namespace boost {
namespace python {
template <typename T>
str dtype_typestr() {
    str s;
    if (std::is_floating_point<T>::value)
        s = "|f";
    else if (std::is_integral<T>::value)
        s = std::is_unsigned<T>::value ? "|u" : "|i";
    else
        throw std::invalid_argument("T must be a builtin arithmetic type");
    s += str(sizeof(T));
    return s;
}
} // python
} // boost

#endif
