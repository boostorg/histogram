// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_STREAMER_HPP_
#define _BOOST_HISTOGRAM_AXIS_STREAMER_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <ostream>
#include <string>

namespace boost {
namespace histogram {

inline std::ostream& operator<<(std::ostream& os, const regular_axis& a)
{
    os << "regular_axis(" << a.bins() << ", " << a[0] << ", " << a[a.bins()];
    if (!a.label().empty())
        os << ", label=" << detail::escape(a.label());
    if (!a.uoflow())
        os << ", uoflow=False";
    os << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const polar_axis& a)
{
    os << "polar_axis(" << a.bins();
    if (a[0] != 0.0)
        os << ", " << a[0];
    if (!a.label().empty())
        os << ", label=" << detail::escape(a.label());
    os << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const variable_axis& a)
{
    os << "variable_axis(" << a[0];
    for (int i = 1; i <= a.bins(); ++i)
        os << ", " << a.left(i);
    if (!a.label().empty())
        os << ", label=" << detail::escape(a.label());
    if (!a.uoflow())
        os << ", uoflow=False";
    os << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const integer_axis& a)
{
    os << "integer_axis(" << a[0] << ", " << a[a.bins() - 1];
    if (!a.label().empty())
        os << ", label=" << detail::escape(a.label());
    if (!a.uoflow())
        os << ", uoflow=False";
    os << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const category_axis& a)
{
    os << "category_axis(";
    for (int i = 0; i < a.bins(); ++i)
        os << detail::escape(a[i])  << (i == (a.bins() - 1)? ")" : ", ");
    return os;
}

}
}

#endif
