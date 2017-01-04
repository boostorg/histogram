// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_
#define _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_

#include <string>
#include <type_traits>
#include <boost/variant/static_visitor.hpp>

namespace boost {
namespace histogram {
namespace detail {

    template <typename T,
              typename = decltype(*std::declval<T&>()),
              typename = decltype(++std::declval<T&>())>
    struct is_iterator {};

    struct linearize : public static_visitor<void> {
        int j;
        std::size_t out = 0, stride = 1;

        void set(int in) { j = in; }

        template <typename A>
        void operator()(const A& a) {
          // the following is highly optimized code that runs in a hot loop;
          // please measure the performance impact of changes
          const int uoflow = a.uoflow();
          // set stride to zero if 'j' is not in range,
          // this communicates the out-of-range condition to the caller
          stride *= (j >= -uoflow) * (j < (a.bins() + uoflow));
          j += (j < 0) * (a.bins() + 2); // wrap around if in < 0
          out += j * stride;
          #pragma GCC diagnostic ignored "-Wstrict-overflow"
          stride *= a.shape();
        }
    };

    struct linearize_x : public static_visitor<void> {
        double x;
        std::size_t out = 0, stride = 1;

        void set(double in) { x = in; }

        template <typename A>
        void
        operator()(const A& a) {
          // j is guaranteed to be in range [-1, bins]
          int j = a.index(x);
          j += (j < 0) * (a.bins() + 2); // wrap around if j < 0
          out += j * stride;
          #pragma GCC diagnostic ignored "-Wstrict-overflow"
          stride *= (j < a.shape()) * a.shape(); // stride == 0 indicates out-of-range
        }
    };

    inline
    std::string escape(const std::string& s) {
        std::string os;
        os.reserve(s.size() + 2);
        os += '\'';
        for (auto sit = s.begin(); sit != s.end(); ++sit) {
            if (*sit == '\'' && (sit == s.begin() || *(sit-1) != '\\'))
                os += "\\\'";
            else
                os += *sit;
        }
        os += '\'';
        return os;
    }

}
}
}

#endif
