// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_
#define _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_

#include <ostream>

namespace boost {
namespace histogram {
namespace detail {

    // assumes that Storage1 and Storage2 have equal size
    template <typename Storage1, typename Storage2>
    bool storage_content_equal(const Storage1& s1, const Storage2& s2)
    {
      for (unsigned i = 0, n = s1.size(); i < n; ++i)
        if (s1.value(i) != s2.value(i) || s1.variance(i) != s2.variance(i))
          return false;
      return true;
    }

    inline
    void escape(std::ostream& os, const char* s) {
        os << '\'';
        for (const char* sit = s; *sit; ++sit) {
            if (*sit == '\'' && (sit == s || *(sit-1) != '\\'))
                os << "\\\'";
            else
                os << *sit;
        }
        os << '\'';
    }

    inline
    void escape(std::ostream& os, const std::string& s) {
        escape(os, s.c_str());
    }

    inline
    bool empty(const char* s) { return !s || s[0] == 0; }

    inline
    bool empty(const std::string& s) { return s.empty(); }
}
}
}

#endif
