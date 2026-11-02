// Copyright 2026 Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_CELL_MAP_HPP
#define BOOST_HISTOGRAM_DETAIL_CELL_MAP_HPP

#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/optional_index.hpp>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

// Maps each cell of an input storage to an offset in an output storage. Each axis has one
// lookup table with one entry per stored input bin, flow bins included. An entry is the
// output offset contributed by that bin. invalid_index skips the input cell and, for an
// outer axis, the whole block of cells below it.
template <class Axes>
class cell_map {
  struct axis_row {
    std::size_t begin;  // first entry of this axis in lut_
    std::size_t extent; // number of stored input bins
    std::size_t stride; // input stride
  };

public:
  explicit cell_map(const Axes& axes) : rows_(make_stack_buffer<axis_row>(axes)) {}

  // Adds the table for the next axis and returns it, to be filled with `extent` entries.
  // The pointer is valid until the next call.
  std::size_t* add_axis(std::size_t extent) {
    assert(naxes_ < rows_.size());
    rows_[naxes_++] = {lut_.size(), extent, in_size_};
    in_size_ *= extent;
    lut_.resize(lut_.size() + extent);
    return lut_.data() + lut_.size() - extent;
  }

  std::size_t input_size() const noexcept { return in_size_; }

  // Calls f(in_offset, out_offset, lut, extent) for each line of cells along the first
  // axis which is not skipped. Cell p of that line goes to out_offset + lut[p].
  template <class F>
  void for_each_line(F&& f) const {
    assert(naxes_ == rows_.size());
    if (naxes_ == 0) {
      // a histogram without axes has one cell
      const std::size_t zero = 0;
      f(std::size_t{0}, std::size_t{0}, &zero, std::size_t{1});
    } else {
      visit(f, 0, 0, naxes_ - 1);
    }
  }

private:
  template <class F>
  void visit(F& f, std::size_t in_off, std::size_t out_off, unsigned iaxis) const {
    const auto& r = rows_[iaxis];
    const std::size_t* const l = lut_.data() + r.begin;
    if (iaxis == 0) {
      f(in_off, out_off, l, r.extent);
      return;
    }
    for (std::size_t p = 0; p < r.extent; ++p)
      if (l[p] != invalid_index)
        visit(f, in_off + p * r.stride, out_off + l[p], iaxis - 1);
  }

  decltype(make_stack_buffer<axis_row>(std::declval<const Axes&>())) rows_;
  std::vector<std::size_t> lut_;
  std::size_t in_size_ = 1;
  unsigned naxes_ = 0;
};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
