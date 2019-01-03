// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ALGORITHM_REDUCE_HPP
#define BOOST_HISTOGRAM_ALGORITHM_REDUCE_HPP

#include <boost/assert.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/histogram/indexed.hpp>
#include <boost/histogram/unsafe_access.hpp>

#include <boost/throw_exception.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace algorithm {

struct reduce_option_type {
  unsigned iaxis = 0;
  double lower, upper;
  unsigned merge = 0;

  reduce_option_type() noexcept = default;

  reduce_option_type(unsigned i, double l, double u, unsigned m)
      : iaxis(i), lower(l), upper(u), merge(m) {
    if (lower == upper)
      BOOST_THROW_EXCEPTION(std::invalid_argument("lower != upper required"));
    if (merge == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("merge > 0 required"));
  }
};

inline reduce_option_type shrink_and_rebin(unsigned iaxis, double lower, double upper,
                                           unsigned merge) {
  return {iaxis, lower, upper, merge};
}

inline reduce_option_type shrink(unsigned iaxis, double lower, double upper) {
  return {iaxis, lower, upper, 1};
}

inline reduce_option_type rebin(unsigned iaxis, unsigned merge) {
  return {iaxis, std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(), merge};
}

/// Convenience overload for single axis.
inline reduce_option_type shrink_and_rebin(double lower, double upper, unsigned merge) {
  return shrink_and_rebin(0, lower, upper, merge);
}

/// Convenience overload for single axis.
inline reduce_option_type shrink(double lower, double upper) {
  return shrink(0, lower, upper);
}

/// Convenience overload for single axis.
inline reduce_option_type rebin(unsigned merge) { return rebin(0, merge); }

template <typename Grid, typename C, typename = detail::requires_iterable<C>>
decltype(auto) reduce(const Grid& grid, const C& options) {
  using axes_type = typename Grid::axes_type;

  struct option_item : reduce_option_type {
    int begin, end;
    bool is_set() const noexcept { return reduce_option_type::merge > 0; }
  };

  auto options_internal = detail::axes_buffer<axes_type, option_item>(grid.rank());
  for (const auto& o : options) {
    auto& oi = options_internal[o.iaxis];
    if (oi.is_set()) // did we already set the option for this axis?
      BOOST_THROW_EXCEPTION(std::invalid_argument("indices must be unique"));
    oi.lower = o.lower;
    oi.upper = o.upper;
    oi.merge = o.merge;
  }

  auto axes = detail::make_empty_axes(unsafe_access::axes(grid));

  unsigned iaxis = 0;
  grid.for_each_axis([&](const auto& a) {
    using T = detail::unqual<decltype(a)>;

    auto& o = options_internal[iaxis];
    o.begin = 0;
    o.end = a.size();
    if (o.is_set()) {
      if (o.lower < o.upper) {
        while (o.begin != o.end && a.value(o.begin) < o.lower) ++o.begin;
        while (o.end != o.begin && a.value(o.end - 1) >= o.upper) --o.end;
      } else if (o.lower > o.upper) {
        // for inverted axis::regular
        while (o.begin != o.end && a.value(o.begin) > o.lower) ++o.begin;
        while (o.end != o.begin && a.value(o.end - 1) <= o.upper) --o.end;
      }
      o.end -= (o.end - o.begin) % o.merge;
      auto a2 = T(a, o.begin, o.end, o.merge);
      axis::get<T>(detail::axis_get(axes, iaxis)) = a2;
    } else {
      o.merge = 1;
      axis::get<T>(detail::axis_get(axes, iaxis)) = a;
    }
    ++iaxis;
  });

  auto result = Grid(std::move(axes), detail::make_default(unsafe_access::storage(grid)));

  detail::axes_buffer<axes_type, int> idx(grid.rank());
  for (auto x : indexed(grid, true)) {
    auto i = idx.begin();
    auto o = options_internal.begin();
    for (auto j : x) {
      *i = (j - o->begin);
      if (*i <= -1)
        *i = -1;
      else {
        *i /= o->merge;
        const int end = (o->end - o->begin) / o->merge;
        if (*i > end) *i = end;
      }
      ++i;
      ++o;
    }
    unsafe_access::add_value(result, idx, *x);
  }

  return result;
}

template <class Grid, class... Ts>
decltype(auto) reduce(const Grid& grid, const reduce_option_type& t, Ts&&... ts) {
  // this must be in one line, because any of the ts could be a temporary
  return reduce(grid, std::initializer_list<reduce_option_type>{t, ts...});
}

} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif
