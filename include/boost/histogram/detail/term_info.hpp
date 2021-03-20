// Copyright 2021 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_TERM_INFO_HPP
#define BOOST_HISTOGRAM_DETAIL_TERM_INFO_HPP

#if defined __has_include
#if __has_include(<sys/ioctl.h>) && __has_include(<unistd.h>)
#include <sys/ioctl.h>
#include <unistd.h>
#else
#include <boost/config/workaround.hpp>
#include <cstdlib>
#endif
#endif

namespace boost {
namespace histogram {
namespace detail {

struct term_info {
  int width = 78;
  bool utf8 = true;

  term_info() {
#if defined TIOCGWINSZ
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    width = w.ws_col;
#else
#if BOOST_WORKAROUND(BOOST_MSVC, >= 0)
    char* buf;
    std::size_t size;
    if (std::_dupenv_s(&buf, &size, "COLUMNS") == 0) { width = std::atoi(buf); }
    std::free(buf);
#else
    if (const char* s = std::getenv("COLUMNS")) width = std::atoi(s);
#endif
#endif
  }
};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
