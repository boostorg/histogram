// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ALGORITHM_REDUCE_HPP
#define BOOST_HISTOGRAM_ALGORITHM_REDUCE_HPP

#include <boost/assert.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/index_mapper.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/unsafe_access.hpp>
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
    if (lower == upper) throw std::invalid_argument("lower != upper required");
    if (merge == 0) throw std::invalid_argument("merge > 0 required");
  }

  operator bool() const noexcept { return merge; }
};

reduce_option_type shrink(unsigned iaxis, double lower, double upper) {
  return {iaxis, lower, upper, 1};
}

reduce_option_type shrink_and_rebin(unsigned iaxis, double lower, double upper,
                                    unsigned merge) {
  return {iaxis, lower, upper, merge};
}

reduce_option_type rebin(unsigned iaxis, unsigned merge) {
  return {iaxis, std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(), merge};
}

template <typename A, typename S, typename C, typename = detail::requires_iterable<C>>
histogram<A, S> reduce(const histogram<A, S>& h, const C& c) {
  auto options =
      boost::container::static_vector<reduce_option_type, axis::limit>(h.rank());
  for (const auto& o : c) {
    auto& opt_ref = options[o.iaxis];
    if (opt_ref) throw std::invalid_argument("indices must be unique");
    opt_ref.lower = o.lower;
    opt_ref.upper = o.upper;
    opt_ref.merge = o.merge;
  }

  auto r_axes = detail::make_empty_axes(unsafe_access::axes(h));

  detail::index_mapper_reduce im(h.rank());
  auto im_iter = im.begin();
  std::size_t stride[2] = {1, 1};
  unsigned iaxis = 0;
  h.for_each_axis([&](const auto& a) {
    using T = detail::unqual<decltype(a)>;

    const auto n = axis::traits::extend(a);
    im.total *= n;
    im_iter->stride[0] = stride[0];
    stride[0] *= n;
    auto set_flow = [im_iter](int i, const auto& a) {
      switch (axis::traits::options(a)) {
        case axis::option_type::overflow:
          im_iter->overflow[i] = a.size();
          im_iter->underflow[i] = -1;
          break;
        case axis::option_type::underflow_and_overflow:
          im_iter->overflow[i] = a.size();
          im_iter->underflow[i] = a.size() + 1;
          break;
        default: im_iter->underflow[i] = -1; im_iter->overflow[i] = -1;
      };
    };
    set_flow(0, a);

    const auto& opt = options[iaxis];
    unsigned begin = 0, end = a.size(), merge = 1;
    if (opt) {
      merge = opt.merge;
      if (opt.lower < opt.upper) {
        while (begin != end && a.value(begin) < opt.lower) ++begin;
        while (end != begin && a.value(end - 1) >= opt.upper) --end;
      } else if (opt.lower > opt.upper) {
        // for inverted axis::regular
        while (begin != end && a.value(begin) > opt.lower) ++begin;
        while (end != begin && a.value(end - 1) <= opt.upper) --end;
      }
      end -= (end - begin) % merge;
      auto a2 = T(a, begin, end, merge);
      axis::get<T>(detail::axis_get(r_axes, iaxis)) = a2;
      im_iter->stride[1] = stride[1];
      stride[1] *= axis::traits::extend(a2);
      set_flow(1, a2);
    } else {
      axis::get<T>(detail::axis_get(r_axes, iaxis)) = a;
      im_iter->stride[1] = stride[1];
      stride[1] *= axis::traits::extend(a);
      set_flow(1, a);
    }
    im_iter->begin = begin;
    im_iter->end = end;
    im_iter->merge = merge;
    ++im_iter;
    ++iaxis;
  });

  auto h_r = histogram<A, S>(
      std::move(r_axes),
      detail::static_if<detail::has_allocator<S>>(
          [&h](auto) { return S(unsafe_access::storage(h).get_allocator()); },
          [](auto) { return S(); }, 0));

  im(unsafe_access::storage(h_r), unsafe_access::storage(h));
  return h_r;
}

template <typename A, typename S, typename... Ts>
histogram<A, S> reduce(const histogram<A, S>& h, const reduce_option_type& t,
                       Ts&&... ts) {
  // this must be in one line, because any of the ts could be a temporary
  return reduce(h, std::initializer_list<reduce_option_type>{t, ts...});
}

} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif
