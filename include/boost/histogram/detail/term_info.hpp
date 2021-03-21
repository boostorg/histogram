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

namespace term_info {
struct env_t {
  char* data;
  std::size_t size = 0;

  env_t(const char* key) {
#if BOOST_WORKAROUND(BOOST_MSVC, >= 0)
    _dupenv_s(&data, &size, key);
#else
    data = std::getenv(key);
    if (data) size = std::strlen(data);
#endif
  }

  ~env_t() {
#if BOOST_WORKAROUND(BOOST_MSVC, >= 0)
    std::free(data);
#endif
  }

  bool contains(const char* s) {
    const std::size_t n = std::strlen(s);
    if (size < n) return false;
    return std::strstr(data, s);
  }

  int to_int() { return size ? std::atoi(data) : 0; }
};

inline bool utf8() {
  env_t env("LANG");
  bool b = true;
  if (env.size) b = env.contains("UTF") || env.contains("utf");
  return b;
} // namespace term_info

inline int width() {
  int w = 78;
#if defined TIOCGWINSZ
  struct winsize ws;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
  w = ws.ws_col;
#else
  env_t env("COLUMNS");
  w = env.to_int();
#endif
  return w;
}
} // namespace term_info

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
