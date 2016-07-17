// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_
#define _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_

#include <string>

namespace boost {
namespace histogram {
namespace detail {

    std::string escape(const std::string& s) {
        std::string os;
        os += '\'';
        for (unsigned i = 0, n = s.size(); i < n; ++i) {
            const char c = s[i];
            if (c == '\'' && (i == 0 || s[i - 1] != '\\'))
                os += "\\\'";
            else
                os += c;
        }
        os += '\'';
        return os;
    }

}
}
}

#endif
