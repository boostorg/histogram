//          Copyright Oliver Kowalke, Hans Dembinski 2014-2022.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_PREFETCH_H
#define BOOST_HISTOGRAM_DETAIL_PREFETCH_H

#include <boost/config.hpp>
#include <cstddef>
#include <cstdint>

#if BOOST_GCC || BOOST_CLANG
#define BOOST_HISTOGRAM_DETAIL_HAVE_BUILTIN_PREFETCH
#elif BOOST_INTEL
#define BOOST_HISTOGRAM_DETAIL_HAVE_MM_PREFETCH
#include <immintrin.h>
#elif BOOST_MSVC && !defined(_M_ARM) && !defined(_M_ARM64)
#define BOOST_HISTOGRAM_DETAIL_HAVE_MM_PREFETCH
#include <immintrin.h>
#endif

namespace boost {
namespace histogram {
namespace detail {

// modern architectures have cachelines with 64byte length
// ARM Cortex-A15 32/64byte, Cortex-A9 16/32/64bytes
// MIPS 74K: 32byte, 4KEc: 16byte
// ist should be safe to use 64byte for all
static constexpr std::size_t cache_alignment{64};
static constexpr std::size_t cacheline_length{64};
// lookahead size for prefetching
static constexpr std::size_t prefetch_stride{4 * cacheline_length};

BOOST_FORCEINLINE
void prefetch(const void* addr) {
#if defined BOOST_HISTOGRAM_DETAIL_HAVE_MM_PREFETCH
  _mm_prefetch((const char*)addr, _MM_HINT_T0);
#elif defined BOOST_HISTOGRAM_DETAIL_HAVE_BUILTIN_PREFETCH
  // L1 cache : hint == 1
  __builtin_prefetch((void*)addr, 1, 1);
#endif
  // no prefetch available
}

#undef BOOST_HISTOGRAM_DETAIL_HAVE_MM_PREFETCH
#undef BOOST_HISTOGRAM_DETAIL_HAVE_BUILTIN_PREFETCH

} // namespace detail
} // namespace histogram
} // namespace boost

#endif // BOOST_HISTOGRAM_DETAIL_PREFETCH_H
