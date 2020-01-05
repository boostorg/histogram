// Copyright 2018-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ALGORITHM_REDUCE_HPP
#define BOOST_HISTOGRAM_ALGORITHM_REDUCE_HPP

#include <boost/assert.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/make_default.hpp>
#include <boost/histogram/detail/static_if.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/histogram/indexed.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/throw_exception.hpp>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace boost {
namespace histogram {
namespace detail {
struct reduce_command {
  static constexpr unsigned unset = static_cast<unsigned>(-1);
  unsigned iaxis;
  union {
    axis::index_type index;
    double value;
  } begin;
  union {
    axis::index_type index;
    double value;
  } end;
  unsigned merge = 0; // default value indicates unset option
  enum class state_t : char {
    rebin,
    slice,
    shrink,
  } state;
  // for internal use by the reduce algorithm
  bool is_ordered;
  bool use_underflow_bin;
  bool use_overflow_bin;
};
} // namespace detail

namespace algorithm {

/** Base type for all @ref reduce commands.

 Use this type to store commands in a container. The internals of this type are an
 implementation detail. Casting a derived command to this base is safe and never causes
 slicing.
 */
using reduce_command = detail::reduce_command;

using reduce_option [[deprecated("use reduce_command instead")]] =
    reduce_command; ///< deprecated

/** Shrink and rebin command to be used in reduce().

  To @ref shrink and @ref rebin in one command (see the respective commands for more
  details). Equivalent to passing both commands for the same axis to @ref reduce.
 */
struct shrink_and_rebin : reduce_command {

  /** Command is applied to axis with given index.

    @param iaxis which axis to operate on.
    @param lower lowest bound that should be kept.
    @param upper highest bound that should be kept. If upper is inside bin interval, the
    whole interval is removed.
    @param merge how many adjacent bins to merge into one.
  */
  shrink_and_rebin(unsigned iaxis, double lower, double upper, unsigned merge) {
    if (lower == upper)
      BOOST_THROW_EXCEPTION(std::invalid_argument("lower != upper required"));
    if (merge == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("merge > 0 required"));
    reduce_command::iaxis = iaxis;
    reduce_command::begin.value = lower;
    reduce_command::end.value = upper;
    reduce_command::merge = merge;
    reduce_command::state = reduce_command::state_t::shrink;
  }

  /** Command is applied to corresponding axis in order of reduce arguments.

    @param lower lowest bound that should be kept.
    @param upper highest bound that should be kept. If upper is inside bin interval, the
    whole interval is removed.
    @param merge how many adjacent bins to merge into one.
  */
  shrink_and_rebin(double lower, double upper, unsigned merge)
      : shrink_and_rebin(reduce_command::unset, lower, upper, merge) {}
};

/** Slice and rebin command to be used in reduce().

  To @ref slice and @ref rebin in one command (see the respective commands for more
  details). Equivalent to passing both commands for the same axis to @ref reduce.
 */
struct slice_and_rebin : reduce_command {

  /** Command is applied to axis with given index.

    @param iaxis which axis to operate on.
    @param begin first index that should be kept.
    @param end one past the last index that should be kept.
    @param merge how many adjacent bins to merge into one.
  */
  slice_and_rebin(unsigned iaxis, axis::index_type begin, axis::index_type end,
                  unsigned merge) {
    if (!(begin < end))
      BOOST_THROW_EXCEPTION(std::invalid_argument("begin < end required"));
    if (merge == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("merge > 0 required"));

    reduce_command::iaxis = iaxis;
    reduce_command::begin.index = begin;
    reduce_command::end.index = end;
    reduce_command::merge = merge;
    reduce_command::state = reduce_command::state_t::slice;
  }

  /** Command is applied to corresponding axis in order of reduce arguments.

    @param begin first index that should be kept.
    @param end one past the last index that should be kept.
    @param merge how many adjacent bins to merge into one.
  */
  slice_and_rebin(axis::index_type begin, axis::index_type end, unsigned merge)
      : slice_and_rebin(reduce_command::unset, begin, end, merge) {}
};

/** Shrink command to be used in reduce().

  The shrink is inclusive. The bin which contains the first value starts the range of bins
  to keep. The bin which contains the second value is the last included in that range.
  When the second value is exactly equal to a lower bin edge, then the previous bin is
  the last in the range.
 */
struct shrink : shrink_and_rebin {

  /** Command is applied to axis with given index.

    @param iaxis which axis to operate on.
    @param lower bin which contains lower is first to be kept.
    @param upper bin which contains upper is last to be kept, except if upper is equal to
    the lower edge.
  */
  shrink(unsigned iaxis, double lower, double upper)
      : shrink_and_rebin{iaxis, lower, upper, 1u} {}

  /** Command is applied to corresponding axis in order of reduce arguments.

    @param lower bin which contains lower is first to be kept.
    @param upper bin which contains upper is last to be kept, except if upper is equal to
    the lower edge.
  */
  shrink(double lower, double upper) : shrink{reduce_command::unset, lower, upper} {}
};

/** Slice command to be used in reduce().

  Slicing works like shrinking, but uses bin indices instead of values.
 */
struct slice : slice_and_rebin {
  /** Command is applied to axis with given index.

    @param iaxis which axis to operate on.
    @param begin first index that should be kept.
    @param end one past the last index that should be kept.
  */
  slice(unsigned iaxis, axis::index_type begin, axis::index_type end)
      : slice_and_rebin{iaxis, begin, end, 1u} {}

  /** Command is applied to corresponding axis in order of reduce arguments.

    @param begin first index that should be kept.
    @param end one past the last index that should be kept.
  */
  slice(axis::index_type begin, axis::index_type end)
      : slice{reduce_command::unset, begin, end} {}
};

/** Rebin command to be used in reduce().

  The command merges N adjacent bins into one. This makes the axis coarser and the bins
  wider. The original number of bins is divided by N. If there is a rest to this devision,
  the axis is implicitly shrunk at the upper end by that rest.
 */
struct rebin : reduce_command {
  /** Command is applied to axis with given index.

    @param iaxis which axis to operate on.
    @param merge how many adjacent bins to merge into one.
  */
  rebin(unsigned iaxis, unsigned merge) {
    if (merge == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("merge > 0 required"));
    reduce_command::iaxis = iaxis;
    reduce_command::merge = merge;
    reduce_command::state = reduce_command::state_t::rebin;
  }

  /** Command is applied to corresponding axis in order of reduce arguments.

    @param merge how many adjacent bins to merge into one.
  */
  rebin(unsigned merge) : rebin{reduce_command::unset, merge} {}
};

/** Shrink, slice, and/or rebin axes of a histogram.

  Returns a new reduced histogram and leaves the original histogram untouched.

  The commands @ref rebin and @ref shrink or @ref slice for the same axis are
  automatically combined, this is not an error. Passing a @ref shrink and a @ref slice
  command for the same axis or two @ref rebin commands triggers an invalid_argument
  exception. It is safe to reduce histograms with some axis that are not reducible along
  the other axes. Trying to reducing a non-reducible axis triggers an invalid_argument
  exception.

  @param hist original histogram.
  @param options iterable sequence of reduce commands: shrink_and_rebin, slice_and_rebin,
  @ref shrink, @ref slice, or @ref rebin. The element type of the iterable should be
  <a href="./boost/histogram/algorithm/reduce_command.html">reduce_command</a>.
 */
template <class Histogram, class Iterable, class = detail::requires_iterable<Iterable>>
Histogram reduce(const Histogram& hist, const Iterable& options) {
  using axis::index_type;

  const auto& old_axes = unsafe_access::axes(hist);

  auto opts = detail::make_stack_buffer<reduce_command>(old_axes);
  unsigned iaxis = 0;
  for (const reduce_command& o_in : options) {
    BOOST_ASSERT(o_in.merge > 0);
    if (o_in.iaxis != reduce_command::unset && o_in.iaxis >= hist.rank())
      BOOST_THROW_EXCEPTION(std::invalid_argument("invalid axis index"));
    auto& o_out = opts[o_in.iaxis == reduce_command::unset ? iaxis : o_in.iaxis];
    if (o_out.merge == 0) {
      o_out = o_in;
    } else {
      // Some option was already set for this axis, see if we can combine requests.
      // We can combine a rebin and non-rebin request.
      if (!((o_in.state == reduce_command::state_t::rebin) ^
            (o_out.state == reduce_command::state_t::rebin)) ||
          (o_out.merge > 1 && o_in.merge > 1))
        BOOST_THROW_EXCEPTION(std::invalid_argument(
            "multiple non-fuseable reduce requests for axis " +
            std::to_string(o_in.iaxis == reduce_command::unset ? iaxis : o_in.iaxis)));
      if (o_in.state != reduce_command::state_t::rebin) {
        o_out.state = o_in.state;
        o_out.begin = o_in.begin;
        o_out.end = o_in.end;
      } else {
        o_out.merge = o_in.merge;
      }
    }
    o_out.iaxis = reduce_command::unset; // value not used below
    ++iaxis;
  }

  // make new axes container with default-constructed axis instances
  auto axes = detail::make_default(old_axes);
  detail::static_if<detail::is_tuple<decltype(axes)>>(
      [](auto&, const auto&) {},
      [](auto& axes, const auto& old_axes) {
        axes.reserve(old_axes.size());
        detail::for_each_axis(old_axes, [&axes](const auto& a) {
          axes.emplace_back(detail::make_default(a));
        });
      },
      axes, old_axes);

  // override default-constructed axis instances with modified instances
  iaxis = 0;
  hist.for_each_axis([&](const auto& a_in) {
    using A = std::decay_t<decltype(a_in)>;
    using AO = axis::traits::get_options<A>;
    auto& o = opts[iaxis];
    o.is_ordered = axis::traits::ordered(a_in);
    if (o.merge > 0) { // option is set?
      detail::static_if_c<axis::traits::is_reducible<A>::value>(
          [&o](auto&& a_out, const auto& a_in) {
            using A = std::decay_t<decltype(a_in)>;
            if (o.state == reduce_command::state_t::rebin) {
              o.begin.index = 0;
              o.end.index = a_in.size();
            } else {
              if (o.state == reduce_command::state_t::shrink) {
                const auto end_value = o.end.value;
                o.begin.index = axis::traits::index(a_in, o.begin.value);
                o.end.index = axis::traits::index(a_in, o.end.value);
                // end = index + 1, unless end_value is exactly equal to (upper) bin edge
                if (axis::traits::value_as<double>(a_in, o.end.index) != end_value)
                  ++o.end.index;
              }
              // limit [begin, end] to [0, size()]
              if (o.begin.index < 0) o.begin.index = 0;
              if (o.end.index > a_in.size()) o.end.index = a_in.size();
            }
            // shorten the index range to a multiple of o.merge;
            // example [1, 4] with merge = 2 is reduced to [1, 3]
            o.end.index -=
                (o.end.index - o.begin.index) % static_cast<index_type>(o.merge);
            a_out = A(a_in, o.begin.index, o.end.index, o.merge);
          },
          [iaxis](auto&&, const auto&) {
            BOOST_THROW_EXCEPTION(std::invalid_argument("axis " + std::to_string(iaxis) +
                                                        " is not reducible"));
          },
          axis::get<A>(detail::axis_get(axes, iaxis)), a_in);
      // will be configurable with crop()
      o.use_underflow_bin = AO::test(axis::option::underflow);
      o.use_overflow_bin = AO::test(axis::option::overflow);
    } else {
      // option was not set for this axis; fill noop values and copy original axis
      o.merge = 1;
      o.begin.index = 0;
      o.end.index = a_in.size();
      axis::get<A>(detail::axis_get(axes, iaxis)) = a_in;
      o.use_underflow_bin = AO::test(axis::option::underflow);
      o.use_overflow_bin = AO::test(axis::option::overflow);
    }
    ++iaxis;
  });

  auto idx = detail::make_stack_buffer<index_type>(axes);
  auto result =
      Histogram(std::move(axes), detail::make_default(unsafe_access::storage(hist)));
  for (auto&& x : indexed(hist, coverage::all)) {
    auto i = idx.begin();
    auto o = opts.begin();
    bool skip = false;

    for (auto j : x.indices()) {
      *i = (j - o->begin.index);
      if (o->is_ordered && *i <= -1) {
        *i = -1;
        if (!o->use_underflow_bin) skip = true;
      } else {
        if (*i >= 0)
          *i /= static_cast<index_type>(o->merge);
        else
          *i = o->end.index;
        const auto reduced_axis_end =
            (o->end.index - o->begin.index) / static_cast<index_type>(o->merge);
        if (*i >= reduced_axis_end) {
          *i = reduced_axis_end;
          if (!o->use_overflow_bin) skip = true;
        }
      }

      ++i;
      ++o;
    }

    if (!skip) result.at(idx) += *x;
  }

  return result;
}

/** Shrink, slice, and/or rebin axes of a histogram.

  Returns a new reduced histogram and leaves the original histogram untouched.

  The commands @ref rebin and @ref shrink or @ref slice for the same axis are
  automatically combined, this is not an error. Passing a @ref shrink and a @ref slice
  command for the same axis or two @ref rebin commands triggers an invalid_argument
  exception. It is safe to reduce histograms with some axis that are not reducible along
  the other axes. Trying to reducing a non-reducible axis triggers an invalid_argument
  exception.

  @param hist original histogram.
  @param opt first reduce command; one of @ref shrink, @ref slice, @ref rebin,
  shrink_and_rebin, or slice_or_rebin.
  @param opts more reduce commands.
 */
template <class Histogram, class... Ts>
Histogram reduce(const Histogram& hist, const reduce_command& opt, const Ts&... opts) {
  // this must be in one line, because any of the ts could be a temporary
  return reduce(hist, std::initializer_list<reduce_command>{opt, opts...});
}

} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif
