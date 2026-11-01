// Copyright 2018-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ALGORITHM_REDUCE_HPP
#define BOOST_HISTOGRAM_ALGORITHM_REDUCE_HPP

#include <algorithm>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/make_default.hpp>
#include <boost/histogram/detail/optional_index.hpp>
#include <boost/histogram/detail/reduce_command.hpp>
#include <boost/histogram/detail/static_if.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/throw_exception.hpp>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

struct reduce_axis_map {
  std::size_t begin;  // first entry of this axis in the lookup table
  std::size_t extent; // number of stored input bins, including flow bins
  std::size_t stride; // input stride of this axis
};

// Innermost axis. Consecutive input cells which map to the same output cell are
// summed in a local accumulator; this breaks the store-to-load chain on the output cell.
template <class In, class Out>
void reduce_fill_inner(std::true_type, const In& in, std::size_t in_off, Out& out,
                       std::size_t out_off, const std::size_t* l, std::size_t extent) {
  std::size_t p = 0;
  while (p < extent) {
    const auto o = l[p];
    std::size_t q = p + 1;
    while (q < extent && l[q] == o) ++q;
    if (o != invalid_index) {
      auto& dst = out[out_off + o];
      if (q == p + 1) {
        dst += in[in_off + p];
      } else {
        auto acc = dst;
        for (; p < q; ++p) acc += in[in_off + p];
        dst = acc;
      }
    }
    p = q;
  }
}

template <class In, class Out>
void reduce_fill_inner(std::false_type, const In& in, std::size_t in_off, Out& out,
                       std::size_t out_off, const std::size_t* l, std::size_t extent) {
  for (std::size_t p = 0; p < extent; ++p)
    if (l[p] != invalid_index) out[out_off + l[p]] += in[in_off + p];
}

// Accumulate input cells into output cells. The lookup table maps each stored input
// position to an output offset; invalid_index skips the input cell and, for an outer
// axis, the whole block of cells below it.
template <class In, class Out>
void reduce_fill(const In& in, std::size_t in_off, Out& out, std::size_t out_off,
                 const std::size_t* lut, const reduce_axis_map* maps, unsigned iaxis) {
  const auto& m = maps[iaxis];
  const std::size_t* const l = lut + m.begin;
  if (iaxis == 0) {
    // local accumulation needs a true reference to an arithmetic cell
    using ref_t = decltype(out[std::size_t{}]);
    using fast_t =
        std::integral_constant<bool, std::is_lvalue_reference<ref_t>::value &&
                                         std::is_arithmetic<std::decay_t<ref_t>>::value>;
    reduce_fill_inner(fast_t{}, in, in_off, out, out_off, l, m.extent);
  } else {
    for (std::size_t p = 0; p < m.extent; ++p)
      if (l[p] != invalid_index)
        reduce_fill(in, in_off + p * m.stride, out, out_off + l[p], lut, maps, iaxis - 1);
  }
}

} // namespace detail

namespace algorithm {

/** Holder for a reduce command.

  Use this type to store reduce commands in a container. The internals of this type are an
  implementation detail.
*/
using reduce_command = detail::reduce_command;

/** Shrink command to be used in `reduce`.

  Command is applied to axis with given index.

  Shrinking is based on an inclusive value interval. The bin which contains the first
  value starts the range of bins to keep. The bin which contains the second value is the
  last included in that range. When the second value is exactly equal to a lower bin edge,
  then the previous bin is the last in the range.

  The counts in removed bins are added to the corresponding underflow and overflow bins,
  if they are present. If they are not present, the counts are discarded. Also see
  `crop`, which always discards the counts.

  @param iaxis which axis to operate on.
  @param lower bin which contains lower is first to be kept.
  @param upper bin which contains upper is last to be kept, except if upper is equal to
  the lower edge.
*/
inline reduce_command shrink(unsigned iaxis, double lower, double upper) {
  if (lower == upper)
    BOOST_THROW_EXCEPTION(std::invalid_argument("lower != upper required"));
  reduce_command r;
  r.iaxis = iaxis;
  r.range = reduce_command::range_t::values;
  r.begin.value = lower;
  r.end.value = upper;
  r.merge = 1;
  r.crop = false;
  return r;
}

/** Shrink command to be used in `reduce`.

  Command is applied to corresponding axis in order of reduce arguments.

  Shrinking is based on an inclusive value interval. The bin which contains the first
  value starts the range of bins to keep. The bin which contains the second value is the
  last included in that range. When the second value is exactly equal to a lower bin edge,
  then the previous bin is the last in the range.

  The counts in removed bins are added to the corresponding underflow and overflow bins,
  if they are present. If they are not present, the counts are discarded. Also see
  `crop`, which always discards the counts.

  @param lower bin which contains lower is first to be kept.
  @param upper bin which contains upper is last to be kept, except if upper is equal to
  the lower edge.
*/
inline reduce_command shrink(double lower, double upper) {
  return shrink(reduce_command::unset, lower, upper);
}

/** Crop command to be used in `reduce`.

  Command is applied to axis with given index.

  Works like `shrink` (see shrink documentation for details), but counts in removed
  bins are always discarded, whether underflow and overflow bins are present or not.

  @param iaxis which axis to operate on.
  @param lower bin which contains lower is first to be kept.
  @param upper bin which contains upper is last to be kept, except if upper is equal to
  the lower edge.
*/
inline reduce_command crop(unsigned iaxis, double lower, double upper) {
  reduce_command r = shrink(iaxis, lower, upper);
  r.crop = true;
  return r;
}

/** Crop command to be used in `reduce`.

  Command is applied to corresponding axis in order of reduce arguments.

  Works like `shrink` (see shrink documentation for details), but counts in removed bins
  are discarded, whether underflow and overflow bins are present or not. If the cropped
  range goes beyond the axis range, then the content of the underflow
  or overflow bin which overlaps with the range is kept.

  If the counts in an existing underflow or overflow bin are discared by the crop, the
  corresponding memory cells are not physically removed. Only their contents are set to
  zero. This technical limitation may be lifted in the future, then crop may completely
  remove the cropped memory cells.

  @param lower bin which contains lower is first to be kept.
  @param upper bin which contains upper is last to be kept, except if upper is equal to
  the lower edge.
*/
inline reduce_command crop(double lower, double upper) {
  return crop(reduce_command::unset, lower, upper);
}

///  Whether to behave like `shrink` or `crop` regarding removed bins.
enum class slice_mode { shrink, crop };

/** Slice command to be used in `reduce`.

  Command is applied to axis with given index.

  Slicing works like `shrink` or `crop`, but uses bin indices instead of values.

  @param iaxis which axis to operate on.
  @param begin first index that should be kept.
  @param end one past the last index that should be kept.
  @param mode whether to behave like `shrink` or `crop` regarding removed bins.
*/
inline reduce_command slice(unsigned iaxis, axis::index_type begin, axis::index_type end,
                            slice_mode mode = slice_mode::shrink) {
  if (!(begin < end))
    BOOST_THROW_EXCEPTION(std::invalid_argument("begin < end required"));

  reduce_command r;
  r.iaxis = iaxis;
  r.range = reduce_command::range_t::indices;
  r.begin.index = begin;
  r.end.index = end;
  r.merge = 1;
  r.crop = mode == slice_mode::crop;
  return r;
}

/** Slice command to be used in `reduce`.

  Command is applied to corresponding axis in order of reduce arguments.

  Slicing works like `shrink` or `crop`, but uses bin indices instead of values.

  @param begin first index that should be kept.
  @param end one past the last index that should be kept.
  @param mode whether to behave like `shrink` or `crop` regarding removed bins.
*/
inline reduce_command slice(axis::index_type begin, axis::index_type end,
                            slice_mode mode = slice_mode::shrink) {
  return slice(reduce_command::unset, begin, end, mode);
}

/** Rebin command to be used in `reduce`.

  Command is applied to axis with given index.

  The command merges N adjacent bins into one. This makes the axis coarser and the bins
  wider. The original number of bins is divided by N. If there is a rest to this devision,
  the axis is implicitly shrunk at the upper end by that rest.

  @param iaxis which axis to operate on.
  @param merge how many adjacent bins to merge into one.
*/
inline reduce_command rebin(unsigned iaxis, unsigned merge) {
  if (merge == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("merge > 0 required"));
  reduce_command r;
  r.iaxis = iaxis;
  r.merge = merge;
  r.range = reduce_command::range_t::none;
  r.crop = false;
  return r;
}

/** Rebin command to be used in `reduce`.

  Command is applied to corresponding axis in order of reduce arguments.

  The command merges N adjacent bins into one. This makes the axis coarser and the bins
  wider. The original number of bins is divided by N. If there is a rest to this devision,
  the axis is implicitly shrunk at the upper end by that rest.

  @param merge how many adjacent bins to merge into one.
*/
inline reduce_command rebin(unsigned merge) {
  return rebin(reduce_command::unset, merge);
}

/** Shrink and rebin command to be used in `reduce`.

  Command is applied to corresponding axis in order of reduce arguments.

  To shrink(unsigned, double, double) and rebin(unsigned, unsigned) in one command (see
  the respective commands for more details). Equivalent to passing both commands for the
  same axis to `reduce`.

  @param iaxis which axis to operate on.
  @param lower lowest bound that should be kept.
  @param upper highest bound that should be kept. If upper is inside bin interval, the
  whole interval is removed.
  @param merge how many adjacent bins to merge into one.
*/
inline reduce_command shrink_and_rebin(unsigned iaxis, double lower, double upper,
                                       unsigned merge) {
  reduce_command r = shrink(iaxis, lower, upper);
  r.merge = rebin(merge).merge;
  return r;
}

/** Shrink and rebin command to be used in `reduce`.

  Command is applied to corresponding axis in order of reduce arguments.

  To `shrink` and `rebin` in one command (see the respective commands for more
  details). Equivalent to passing both commands for the same axis to `reduce`.

  @param lower lowest bound that should be kept.
  @param upper highest bound that should be kept. If upper is inside bin interval, the
  whole interval is removed.
  @param merge how many adjacent bins to merge into one.
*/
inline reduce_command shrink_and_rebin(double lower, double upper, unsigned merge) {
  return shrink_and_rebin(reduce_command::unset, lower, upper, merge);
}

/** Crop and rebin command to be used in `reduce`.

  Command is applied to axis with given index.

  To `crop` and `rebin` in one command (see the respective commands for more
  details). Equivalent to passing both commands for the same axis to `reduce`.

  @param iaxis which axis to operate on.
  @param lower lowest bound that should be kept.
  @param upper highest bound that should be kept. If upper is inside bin interval,
  the whole interval is removed.
  @param merge how many adjacent bins to merge into one.
*/
inline reduce_command crop_and_rebin(unsigned iaxis, double lower, double upper,
                                     unsigned merge) {
  reduce_command r = crop(iaxis, lower, upper);
  r.merge = rebin(merge).merge;
  return r;
}

/** Crop and rebin command to be used in `reduce`.

  Command is applied to corresponding axis in order of reduce arguments.

  To `crop` and `rebin` in one command (see the respective commands for more
  details). Equivalent to passing both commands for the same axis to `reduce`.

  @param lower lowest bound that should be kept.
  @param upper highest bound that should be kept. If upper is inside bin interval,
  the whole interval is removed.
  @param merge how many adjacent bins to merge into one.
*/
inline reduce_command crop_and_rebin(double lower, double upper, unsigned merge) {
  return crop_and_rebin(reduce_command::unset, lower, upper, merge);
}

/** Slice and rebin command to be used in `reduce`.

  Command is applied to axis with given index.

  To `slice` and `rebin` in one command (see the respective commands for more
  details). Equivalent to passing both commands for the same axis to `reduce`.

  @param iaxis which axis to operate on.
  @param begin first index that should be kept.
  @param end one past the last index that should be kept.
  @param merge how many adjacent bins to merge into one.
  @param mode slice mode, see slice_mode.
*/
inline reduce_command slice_and_rebin(unsigned iaxis, axis::index_type begin,
                                      axis::index_type end, unsigned merge,
                                      slice_mode mode = slice_mode::shrink) {
  reduce_command r = slice(iaxis, begin, end, mode);
  r.merge = rebin(merge).merge;
  return r;
}

/** Slice and rebin command to be used in `reduce`.

  Command is applied to corresponding axis in order of reduce arguments.

  To `slice` and `rebin` in one command (see the respective commands for more
  details). Equivalent to passing both commands for the same axis to `reduce`.

  @param begin first index that should be kept.
  @param end one past the last index that should be kept.
  @param merge how many adjacent bins to merge into one.
  @param mode slice mode, see slice_mode.
*/
inline reduce_command slice_and_rebin(axis::index_type begin, axis::index_type end,
                                      unsigned merge,
                                      slice_mode mode = slice_mode::shrink) {
  return slice_and_rebin(reduce_command::unset, begin, end, merge, mode);
}

/** Pick command to be used in `reduce`.

  Command is applied to axis with given index.

  Picking selects an arbitrary subset of bins by index. The new axis consists of the
  picked bins in the order in which the indices are given, which may differ from their
  order in the original axis. In contrast to `slice`, the picked bins do not have to be
  adjacent. Each index must be valid and may only appear once.

  Picking only works on axes that are not ordered, like the category axis, since
  removing an arbitrary subset of bins from an ordered axis would create gaps in the
  axis range. The counts in bins that were not picked are added to the overflow bin,
  if it is present. If it is not present, the counts are discarded. In crop mode, the
  counts in unpicked bins and in the original overflow bin are always discarded.

  @param iaxis which axis to operate on.
  @param indices indices of the bins to keep, must be unique.
  @param mode whether to behave like `shrink` or `crop` regarding removed bins.
*/
inline reduce_command pick(unsigned iaxis, std::vector<axis::index_type> indices,
                           slice_mode mode = slice_mode::shrink) {
  if (indices.empty())
    BOOST_THROW_EXCEPTION(std::invalid_argument("at least one index required"));
  for (auto it = indices.begin(); it != indices.end(); ++it)
    if (std::find(indices.begin(), it, *it) != it)
      BOOST_THROW_EXCEPTION(std::invalid_argument("indices must be unique"));
  reduce_command r;
  r.iaxis = iaxis;
  r.range = reduce_command::range_t::indices_list;
  r.indices = std::move(indices);
  r.merge = 1;
  r.crop = mode == slice_mode::crop;
  return r;
}

/** Pick command to be used in `reduce`.

  Command is applied to corresponding axis in order of reduce arguments.

  Picking selects an arbitrary subset of bins by index, see
  pick(unsigned, std::vector<axis::index_type>, slice_mode) for details.

  @param indices indices of the bins to keep, must be unique.
  @param mode whether to behave like `shrink` or `crop` regarding removed bins.
*/
inline reduce_command pick(std::vector<axis::index_type> indices,
                           slice_mode mode = slice_mode::shrink) {
  return pick(reduce_command::unset, std::move(indices), mode);
}

/** Shrink, crop, slice, pick, and/or rebin axes of a histogram.

  Returns a new reduced histogram and leaves the original histogram untouched.

  The commands `rebin` and `shrink` or `slice` for the same axis are
  automatically combined, this is not an error. Passing a `shrink` and a `slice`
  command for the same axis or two `rebin` commands triggers an `invalid_argument`
  exception. The `pick` command cannot be combined with any other command for the
  same axis. Trying to reducing a non-reducible axis triggers an `invalid_argument`
  exception. Histograms with  non-reducible axes can still be reduced along the
  other axes that are reducible.

  An overload allows one to pass reduce_command as positional arguments.

  @param hist original histogram.
  @param options iterable sequence of reduce commands: `shrink`, `slice`, `rebin`,
  `pick`, `shrink_and_rebin`, or `slice_and_rebin`. The element type of the iterable
  should be `reduce_command`.
*/
template <class Histogram, class Iterable, class = detail::requires_iterable<Iterable>>
Histogram reduce(const Histogram& hist, const Iterable& options) {
  using axis::index_type;

  const auto& old_axes = unsafe_access::axes(hist);
  auto opts = detail::make_stack_buffer(old_axes, reduce_command{});
  detail::normalize_reduce_commands(opts, options);

  auto axes =
      detail::axes_transform(old_axes, [&opts](std::size_t iaxis, const auto& a_in) {
        using A = std::decay_t<decltype(a_in)>;
        using AO = axis::traits::get_options<A>;
        auto& o = opts[iaxis];
        o.is_ordered = axis::traits::ordered(a_in);
        if (o.merge > 0) { // option is set?
          o.use_underflow_bin = AO::test(axis::option::underflow);
          o.use_overflow_bin = AO::test(axis::option::overflow);
          if (o.range == reduce_command::range_t::indices_list) {
            for (const auto idx : o.indices)
              if (idx < 0 || idx >= a_in.size())
                BOOST_THROW_EXCEPTION(std::invalid_argument("index out of range"));
            // crop discards counts of unpicked bins instead of moving them to overflow
            if (o.crop) o.use_overflow_bin = false;
            return detail::static_if_c<axis::traits::has_pick_constructor<A>::value>(
                [&o](const auto& a_in) {
                  static_assert(!axis::traits::is_ordered<A>::value,
                                "pick requires an unordered axis");
                  // lut maps old index to new index; unpicked bins and the
                  // overflow bin map to the overflow bin of the new axis
                  o.reduced_end = static_cast<index_type>(o.indices.size());
                  o.lut.assign(static_cast<std::size_t>(a_in.size()) + 1, o.reduced_end);
                  for (index_type k = 0; k < o.reduced_end; ++k) o.lut[o.indices[k]] = k;
                  return std::decay_t<decltype(a_in)>(
                      a_in, o.indices.data(), o.indices.data() + o.indices.size());
                },
                [iaxis](const auto& a_in) -> std::decay_t<decltype(a_in)> {
                  BOOST_THROW_EXCEPTION(std::invalid_argument(
                      "axis " + std::to_string(iaxis) + " is not pickable"));
                },
                a_in);
          }
          return detail::static_if_c<axis::traits::is_reducible<A>::value>(
              [&o](const auto& a_in) {
                if (o.range == reduce_command::range_t::none) {
                  // no range restriction, pure rebin
                  o.begin.index = 0;
                  o.end.index = a_in.size();
                } else {
                  // range striction, convert values to indices as needed
                  if (o.range == reduce_command::range_t::values) {
                    const auto end_value = o.end.value;
                    o.begin.index = axis::traits::index(a_in, o.begin.value);
                    o.end.index = axis::traits::index(a_in, o.end.value);
                    // end = index + 1, unless end_value equal to upper bin edge
                    if (axis::traits::value_as<double>(a_in, o.end.index) != end_value)
                      ++o.end.index;
                  }

                  // crop flow bins if index range does not include them
                  if (o.crop) {
                    o.use_underflow_bin &= o.begin.index < 0;
                    o.use_overflow_bin &= o.end.index > a_in.size();
                  }

                  // now limit [begin, end] to [0, size()]
                  if (o.begin.index < 0) o.begin.index = 0;
                  if (o.end.index > a_in.size()) o.end.index = a_in.size();
                }
                // shorten the index range to a multiple of o.merge;
                // example [1, 4] with merge = 2 is reduced to [1, 3]
                o.end.index -=
                    (o.end.index - o.begin.index) % static_cast<index_type>(o.merge);
                o.reduced_end =
                    (o.end.index - o.begin.index) / static_cast<index_type>(o.merge);
                using A = std::decay_t<decltype(a_in)>;
                return A(a_in, o.begin.index, o.end.index, o.merge);
              },
              [iaxis](const auto& a_in) {
                return BOOST_THROW_EXCEPTION(std::invalid_argument(
                           "axis " + std::to_string(iaxis) + " is not reducible")),
                       a_in;
              },
              a_in);
        } else {
          // command was not set for this axis; fill noop values and copy original axis
          o.use_underflow_bin = AO::test(axis::option::underflow);
          o.use_overflow_bin = AO::test(axis::option::overflow);
          o.merge = 1;
          o.begin.index = 0;
          o.end.index = a_in.size();
          o.reduced_end = a_in.size();
          return a_in;
        }
      });

  auto result =
      Histogram(std::move(axes), detail::make_default(unsafe_access::storage(hist)));

  // one table per axis maps each stored input position to an output offset
  const auto rank = detail::axes_rank(old_axes);
  std::size_t lut_size = 0;
  detail::for_each_axis(old_axes, [&lut_size](const auto& a) {
    lut_size += static_cast<std::size_t>(axis::traits::extent(a));
  });
  // small tables live on the stack to keep reduce of small histograms cheap
  std::size_t lut_stack[256];
  std::vector<std::size_t> lut_heap;
  std::size_t* lut = lut_stack;
  if (lut_size > 256) {
    lut_heap.resize(lut_size);
    lut = lut_heap.data();
  }
  auto maps = detail::make_stack_buffer<detail::reduce_axis_map>(old_axes);
  {
    std::size_t* l = lut;
    std::size_t in_stride = 1;
    std::size_t out_stride = 1;
    unsigned iaxis = 0;
    detail::for_each_axis(old_axes, [&](const auto& a_in) {
      using A = std::decay_t<decltype(a_in)>;
      using AO = axis::traits::get_options<A>;
      const index_type u = AO::test(axis::option::underflow);
      const index_type ov = AO::test(axis::option::overflow);
      const auto& o = opts[iaxis];
      const auto size = a_in.size();
      auto& m = maps[iaxis];
      m.begin = static_cast<std::size_t>(l - lut);
      m.extent = static_cast<std::size_t>(size + u + ov);
      m.stride = in_stride;
      for (index_type j = -u; j < size + ov; ++j) {
        index_type i;
        bool skip = false;
        if (o.range == reduce_command::range_t::indices_list) {
          i = j < 0 ? o.reduced_end : o.lut[j];
        } else {
          i = (j - o.begin.index);
          if (o.is_ordered && i <= -1) {
            i = -1;
            if (!o.use_underflow_bin) skip = true;
          } else if (i >= 0)
            i /= static_cast<index_type>(o.merge);
          else
            i = o.reduced_end;
        }
        if (i >= o.reduced_end) {
          i = o.reduced_end;
          if (!o.use_overflow_bin) skip = true;
        }
        *l++ =
            skip ? detail::invalid_index : static_cast<std::size_t>(i + u) * out_stride;
      }
      in_stride *= m.extent;
      out_stride *= static_cast<std::size_t>(o.reduced_end + u + ov);
      ++iaxis;
    });
    assert(static_cast<std::size_t>(l - lut) == lut_size);
    assert(in_stride == hist.size());
    assert(out_stride == result.size());
  }

  detail::reduce_fill(unsafe_access::storage(hist), 0, unsafe_access::storage(result), 0,
                      lut, maps.data(), rank - 1);

  return result;
}

/** Shrink, crop, slice, pick, and/or rebin axes of a histogram.

  Returns a new reduced histogram and leaves the original histogram untouched.

  The commands `rebin` and `shrink` or `slice` for the same axis are
  automatically combined, this is not an error. Passing a `shrink` and a `slice`
  command for the same axis or two `rebin` commands triggers an invalid_argument
  exception. The `pick` command cannot be combined with any other command for the
  same axis. It is safe to reduce histograms with some axis that are not reducible along
  the other axes. Trying to reducing a non-reducible axis triggers an invalid_argument
  exception.

  An overload allows one to pass an iterable of reduce_command.

  @param hist original histogram.
  @param opt first reduce command; one of `shrink`, `slice`, `rebin`, `pick`,
  `shrink_and_rebin`, or `slice_or_rebin`.
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
