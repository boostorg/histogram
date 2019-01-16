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

struct reduce_option {
  unsigned iaxis = 0;
  double lower, upper;
  unsigned merge = 0;

  reduce_option() noexcept = default;

  reduce_option(unsigned i, double l, double u, unsigned m)
      : iaxis(i), lower(l), upper(u), merge(m) {
    if (lower == upper)
      BOOST_THROW_EXCEPTION(std::invalid_argument("lower != upper required"));
    if (merge == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("merge > 0 required"));
  }
};

inline reduce_option shrink_and_rebin(unsigned iaxis, double lower, double upper,
                                      unsigned merge) {
  return {iaxis, lower, upper, merge};
}

inline reduce_option shrink(unsigned iaxis, double lower, double upper) {
  return {iaxis, lower, upper, 1};
}

inline reduce_option rebin(unsigned iaxis, unsigned merge) {
  return {iaxis, std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(), merge};
}

/// Convenience overload for single axis.
inline reduce_option shrink_and_rebin(double lower, double upper, unsigned merge) {
  return shrink_and_rebin(0, lower, upper, merge);
}

/// Convenience overload for single axis.
inline reduce_option shrink(double lower, double upper) {
  return shrink(0, lower, upper);
}

/// Convenience overload for single axis.
inline reduce_option rebin(unsigned merge) { return rebin(0, merge); }

template <class Histogram, class C, class = detail::requires_iterable<C>>
decltype(auto) reduce(const Histogram& h, const C& options) {
  const auto& old_axes = unsafe_access::axes(h);

  struct option_item : reduce_option {
    int begin, end;
  };

  auto opts = detail::make_stack_buffer<option_item>(old_axes);
  for (const auto& o : options) {
    auto& oi = opts[o.iaxis];
    if (oi.merge > 0) // did we already set the option for this axis?
      BOOST_THROW_EXCEPTION(std::invalid_argument("indices must be unique"));
    oi.lower = o.lower;
    oi.upper = o.upper;
    oi.merge = o.merge;
  }

  auto axes = detail::static_if<detail::is_tuple<detail::naked<decltype(old_axes)>>>(
      [](const auto& c) { return detail::naked<decltype(c)>(); },
      [](const auto& c) {
        using A = detail::naked<decltype(c)>;
        auto axes = A(c.get_allocator());
        axes.reserve(c.size());
        detail::for_each_axis(c, [&axes](const auto& a) {
          using U = detail::naked<decltype(a)>;
          axes.emplace_back(U());
        });
        return axes;
      },
      old_axes);

  unsigned iaxis = 0;
  h.for_each_axis([&](const auto& a) {
    using T = detail::naked<decltype(a)>;

    auto& o = opts[iaxis];
    o.begin = 0;
    o.end = a.size();
    if (o.merge > 0) { // option is set?
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

  auto storage = detail::make_default(unsafe_access::storage(h));
  auto result = Histogram(std::move(axes), std::move(storage));

  auto idx = detail::make_stack_buffer<int>(unsafe_access::axes(result));
  for (auto x : indexed(h, coverage::all)) {
    auto i = idx.begin();
    auto o = opts.begin();
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
    result.at(idx) += *x;
  }

  return result;
}

template <class Histogram, class... Ts>
decltype(auto) reduce(const Histogram& h, const reduce_option& t, Ts&&... ts) {
  // this must be in one line, because any of the ts could be a temporary
  return reduce(h, std::initializer_list<reduce_option>{t, ts...});
}

} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif
