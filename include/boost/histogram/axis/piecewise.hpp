// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_PIECEWISE_HPP
#define BOOST_HISTOGRAM_AXIS_PIECEWISE_HPP

#include <boost/core/nvp.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/axis/metadata_base.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/detail/convert_integer.hpp>
#include <boost/histogram/detail/relaxed_equal.hpp>
#include <boost/histogram/detail/replace_type.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11/utility.hpp>
#include <boost/throw_exception.hpp>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

namespace axis {

/**Pieces***

  Each piece of a piecewise axis has a start point x0 and an end point x1.

    |--------------------------------------|
    x0                                     x1

  Each piece has n bins. There are three options for varying bin sizes.

      Constant                |------|------|------|------|
       (bi+1 = bi)                     bi     bi+1

      Multiply                |---|-------|---------------|
       (bi+1 = bi * constant)       bi      bi+1

      Add                     |---|-----|-------|---------|
       (bi+1 = bi + constant)       bi    bi+1

***Creating a Piece***

  To create a piece, one starts at a point x0. Starting with a bin size b0, bins are
  added according to the bin spacing option until a termination condition is met. The
  termination condition can be either a bin number n_bins or stopping point x_stop.
  These two possibilities are shown below.

  Add n bins
    Start at x0 = 1. Start with a bin size of b0 = 1. Double the bin size until 4 bins
    are added.
     |-|---|-------|---------------|
   0   2   4   6   8   10  12  14  16

 Add until the piece contains x_stop
   Start at x0 = 1. Start with a bin size of b0 = 3. Add bins of constant size until
   the piece contains x_stop = 14.
     |-----|-----|-----|-----|--*--|
   0   2   4   6   8   10  12  14  16

***Extrapolation and Attachment***

  Pieces can be added to the right or left side of an existing piece. Pieces are
  added with one of two approaches: extrapolation or attachment. Examples of each are
  shown below.
  Extrapolate 2 bins to the right side, doubling the bin size each time.
     0   2   4   6   8
     |---|---|---|---|
                      <- - - New piece - - ->
     |---|---|---|---|-------|---------------|
     0   2   4   6   8   10  12  14  16  18  20
  Attach a new piece with a uniform bin spacing to the right side.
     0   2   4   6   8
     |---|---|---|---|
                      <- - - New piece - - ->
     |---|---|---|---|-|-|-|-|-|-|-|-|-|-|-|-|
     0   2   4   6   8   10  12  14  16  18  20

***Syntax***

using P = piecewise<double>;

auto pa = P(                           // Create a new piece
            P::x0(1),                    // Start at x0 = 1
            P::b0(1),                    // Start with a bin size of b0 = 1
            P::bin_trans::uniform(1),    // Add bins of constant size 1
            P::stop::n_bins(4));         // Add 4 bins

pa.add_right(                          // Add to the right side
             P::bin_trans::multiply(2),  // Extrapolate, doubling the bin size each time
             P::stop::x(16));            // Stop when the piece contains x = 16

pa.add_left(                           // Add to the left side
            P::b0(1),                    // Start with a bin size of b0 = 1
            P::bin_trans::add(1),        // Add bins of constant size 1
            P::stop::x(4));              // Stop when the piece contains x = 4
*/

/// Struct for starting point
class x0 {
public:
  x0(double value) : value_{value} {
    if (!std::isfinite(value))
      BOOST_THROW_EXCEPTION(std::invalid_argument("x0 must be finite"));
  }

  double value() const { return value_; }

private:
  double value_;
};

/// Struct for bin size b0
class b0 {
public:
  b0(double value) : value_{value} {
    if (!(0 < value)) BOOST_THROW_EXCEPTION(std::invalid_argument("b0 must be > 0"));
  }

  double value() const { return value_; }

private:
  double value_;
};

/// Class for bin spacing option and value
class bin_transion {
public:
  enum class option { uniform, multiply, add };

  double calc_next_bin_size(double b) const {
    assert(0 < b);
    switch (option_) {
      case option::uniform: assert(b == value_); return b;
      case option::multiply: return b * value_;
      case option::add: return b + value_;
    }
  }

  bin_transion uniform(double value) = bin_transion{option::uniform, value};
  bin_transion multiply(double value) = bin_transion{option::multiply, value};
  bin_transion add(double value) = bin_transion{option::add, value};

  bool is_uniform() const { return option_ == option::uniform; }
  bool is_multiply() const { return option_ == option::multiply; }
  bool is_add() const { return option_ == option::add; }

  double value() const { return value_; }

private:
  bin_transion(option option, double value) : option_{option}, value_{value} {}

  option option_;
  double value_;
};

/// Class for stopping condition and value
class bin_stop {
public:
  enum class option { n_bins, x };

  bin_stop n_bins(double value) = bin_stop{option::n_bins, value};
  bin_stop x(double value) = bin_stop{option::x, value};

  bool is_n_bins() const { return option_ == option::n_bins; }
  bool is_x() const { return option_ == option::x; }

  double value() const { return value_; }

private:
  bin_stop(option option, double value) : option_{option}, value_{value} {}

  option option_;
  double value_;
};

///
template <class Value>
class piece {
  using value_type = Value;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  piece(x0 x0, b0 b0, bin_trans bin_trans, bin_stop stop, bool is_right)
      : x0_{x0.value}, b0_{b0.value}, bin_trans_{bin_trans} {
    // Assert is handled case
    if (stop.is_n_bins()) {
      if (stop.n_bins.value <= 0) {
        BOOST_THROW_EXCEPTION(std::invalid_argument("n_bins must be > 0"));
      }
    } else if (stop.is_x()) {
      if (stop.x.value <= x0_) {
        BOOST_THROW_EXCEPTION(std::invalid_argument("x must be > x0"));
      }
    }

    // Initialize
    initialize(stop);
  }

  /// Add bins left
  void add_left_bins(index_type n) noexcept {
    assert(0 < n); // TODO: remove
    n = std::abs(n);

    // Update x0 and bin size
    x0_ = reverse(-n);
    b0_ = reverse(-n + 1) - x0_;
    size_++;
  }

  /// Add bins right
  void add_bins_right(index_type n) noexcept {
    assert(0 < n); // TODO: remove
    n = std::abs(n);

    // Update x0 and bin size
    xN_ = reverse(size() + n);
    b0_ = xN_ - reverse(size() + n - 1);
    size_++;
  }

  /// Shifts the axis
  void shift_axis(index_type n) {
    if (n < 0) {
      // Shift axis to the left by n bins
      const auto stop = min_ + delta_;
      min_ += n * (delta_ / size());
      delta_ = stop - min_;
      size_ -= n;

    } else if (0 < n) {
      // Shift axis to the right by n bins
      delta_ /= size();
      delta_ *= size() + n;
      size_ += n;

      x0_ += n * b0_;
    }
  }

  template <class T>
  T forward(T x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    switch (option_) {
      case option::uniform: return forward_uniform(x);
      case option::multiply: return forward_multiply(x);
      case option::add: return forward_add(x);
    }
  }

  index_type size() const noexcept { return size_; }

private:
  // Initialize
  //   Have: (x0, b0, n_bins)
  //     Calc: xN

  // Initialize
  //   Have: (x0, xN, n_bins)
  //     Calc: b0

  // Add right
  //   Have: (x0, b0, n_bins++)
  //     Update: xN

  // Add left
  //   Have: (xN, b0, n_bins++)
  //     Calc: b0
  //     Update: x0

  void initialize(bin_stop stop) {
    xN_ = x0_;
    size_ = 0;
    double b = b0_;

    while (!see_if_need_more_bins()) {
      b = bin_trans_.calc_next_bin_size(b);
      xN_ += b;
      ++size_;
    }
  }

  bool see_if_need_more_bins() {
    if (stop_.is_n_bins()) {
      return size_ >= stop_.n_bins.value;
    } else if (stop_.is_x()) {
      return xN_ >= stop_.x.value;
    }
  }

  // Uniform spacing
  //
  //       b0     b0     b0     b0
  //   N=0    N=1    N=2    N=3    N=4
  //    |------|------|------|------|
  //    x0                          x4
  //
  // The sequence
  //   x0 = x0
  //   x1 = x0 + b0
  //   x2 = x0 + b0 * 2
  //   x3 = x0 + b0 * 3
  //   ...
  //   x  = x0 + b0 * N
  //
  // Formula for N in terms of x
  //   N = (x - x0) / b0
  //
  template <class T>
  interval_value_type forward_linear(T x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    return ((x / unit_type{}) - x0_) / b0_;
  }

  // Multiply spacing
  //
  //   b0  b0*r     b0*r²         b0*r³
  // N=0 N=1   N=2         N=3                N=4
  //  |---|-----|-----------|------------------|
  //  x0                                      x4
  //
  // The sequence (for some value of ψ)
  //   x0 = 1  * b0 / (r - 1) + ψ
  //   x1 = r  * b0 / (r - 1) + ψ
  //   x2 = r² * b0 / (r - 1) + ψ
  //   x3 = r³ * b0 / (r - 1) + ψ
  //   ...
  //   x  = rᴺ * b0 / (r - 1) + ψ
  //
  // Note: the first bin spacing is b0
  //   x1 - x0 = r * b0 / (r - 1) - 1 * b0 / (r - 1)
  //           = (r - 1) * b0 / (r - 1)
  //           = b0
  //
  // Find a formula for N
  //   x - x0   = b0 / (r - 1) * rᴺ - b0 / (r - 1) * 1
  //            = b0 / (r - 1) * (rᴺ - 1)
  //
  // N in terms of x
  //   N = log2(((x - x0) / b0) * (r - 1) + 1) / log2(r)
  //
  template <class T>
  interval_value_type forward_multiply(T x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    const auto z = forward_linear(x); // z = (x - x0) / b0
    const auto r = bin_trans_.value();
    return std::log2(z * (r - 1) + 1) / std::log2(r);
  }

  // Add spacing
  //
  //   b0  b0+r     b0+2r         b0+3r
  // N=0 N=1   N=2         N=3                N=4
  //  |---|-----|-----------|------------------|
  //  x0                                      x4
  //
  // The sequence
  //   x0 = x0
  //   x1 = x0 + 1 * b0
  //   x2 = x0 + 2 * b0 + r
  //   x3 = x0 + 3 * b0 + r * (1 + 2)
  //   x4 = x0 + 4 * b0 + r * (1 + 2 + 3)
  //   ...
  //   x  = x0 + N * b0 + r * (1 + 2 + ... + N-1)
  //      = x0 + N * b0 + r * (N * (N - 1) / 2)
  //
  // N in terms of x
  //            2 * (x - x0) / b0 = N * (N - 1)
  //   N² - N - 2 * (x - x0) / b0 = 0
  // Solve quadratic equation and take positive root
  //
  template <class T>
  interval_value_type forward_multiply(T x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    const auto z = forward_linear(x); // z = (x - x0) / b0
    const auto r = bin_trans_.value();
    return (1 + std::sqrt(1 + 8 * z / r)) / 2; // Check copilot suggestion
  }

  internal_value_type x0_ { 0 }
  double b0_{1};
  bin_trans bin_trans_{bin_trans::uniform{1}};

  internal_value_type xN_{0} index_type size_{0};
};

///
class piecewise {
public:
  /// Construct a piecewise axis with an initial piece.
  piecewise(x0 x0, b0 b0, bin_trans bin_trans, bin_stop stop) {
    //
    const auto p = piece(x0, b0, bin_trans, stop, true);
    v_pieces_.push_back(p);
  }

  ///
  template <class T>
  T forward(T x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    index_type offset = 0;
    const int n_pieces = v_pieces_.size();
    for (int j = 0; j < n_pieces; ++j) {
      const auto& p = v_pieces_[j];
      if (x < p.x1 || j == n_pieces - 1) return offset + p.index(x);
    }
  }

  ///
  template <class ValueType>
  index_type index(ValueType x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    const auto i = forward(x);
    if (i < size_) {
      if (0 <= i)
        return static_cast<index_type>(i); // 0 <= i < size
      else
        return -1; // i < 0
    }
    // upper edge of last bin is inclusive if overflow bin is not present
    if (!options_type::test(option::overflow) && z == 1) return size() - 1;
    return size(); // also returned if x is NaN
  }

  /// Shifts the axis
  void shift_axis(index_type n) {
    if (n < 0) {
      v_pieces_.front().shift_axis(n);
    } else if (0 < n) {
      // Get last element of v_pieces
      v_pieces_.back().shift_axis(n);
    }
  }

  /// Returns index and shift (if axis has grown) for the passed argument.
  std::pair<index_type, index_type> update(value_type x) noexcept {
    assert(options_type::test(option::growth));
    const auto i = forward(x);
    if (i < size()) {
      if (0 <= i) {
        const auto i_int = static_cast<axis::index_type>(i);
        return {i_int, 0};
      } else
        (i != -std::numeric_limits<internal_value_type>::infinity()) {
          const auto i_int = static_cast<axis::index_type>(std::floor(i));
          shift_axis(i_int);
          return {0, -i_int};
        }
      else { // i is -infinity
        return {-1, 0};
      }
    }
    // i either beyond range, infinite, or NaN
    if (z < std::numeric_limits<internal_value_type>::infinity()) {
      const auto i_int = static_cast<axis::index_type>(i);
      const auto n = i_int - size() + 1;
      shift_axis(n);
      return {i, -n};
    }
    // z either infinite or NaN
    return {size(), 0};
  }

  add_left(bin_trans bin_trans, bin_stop stop) {
    double b0 = get_left_b0();
    add_left(b0, bin_trans, stop);
  }
  add_right(bin_trans bin_trans, bin_stop stop) {
    double b0 = get_right_b0();
    add_right(b0, bin_trans, stop);
  }
  add_left(b0 b0, bin_trans bin_trans, bin_stop stop) {
    double x0 = get_left_x();
    const auto p = piece(x0, b0, bin_trans, stop, false);
    v_pieces_.insert(v_pieces_.begin(), p);
  }
  add_right(b0 b0, bin_trans bin_trans, bin_stop stop) {
    double x0 = get_right_x();
    const auto p = piece(x0, b0, bin_trans, stop, true);
    v_pieces_.push_back(p);
  }

private:
  double get_left_x() const {
    assert(!v_pieces_.empty());
    return v_pieces_.front().x0;
  }

  double get_right_x() const {
    assert(!v_pieces_.empty());
    return v_pieces_.back().x1;
  }

  /// Forward transform of external value x.
  template <class T>
  auto forward(T x) const {
    return std::pow(x, power);
  }

  /// Inverse transform of internal value x.
  template <class T>
  auto inverse(T x) const {
    return std::pow(x, 1.0 / power);
  }

  std::vector<piece> v_pieces_{};
};

/** Axis for unit intervals on the real line.

  The most common binning strategy. Very fast. Binning is a O(1) operation.

  If the axis has an overflow bin (the default), a value on the upper edge of the last
  bin is put in the overflow bin. The axis range represents a semi-open interval.

  If the overflow bin is deactivated, then a value on the upper edge of the last bin is
  still counted towards the last bin. The axis range represents a closed interval.

  The options `growth` and `circular` are mutually exclusive.

  @tparam Value input value type, must be floating point.
  @tparam Transform builtin or user-defined transform type.
  @tparam MetaData type to store meta data.
  @tparam Options see boost::histogram::axis::option.
 */
template <class Value, class MetaData, class Options>
class unit_regular
    : public iterator_mixin<unit_regular<Value, Transform, MetaData, Options>>,
      protected detail::replace_default<Transform, transform::id>,
      public metadata_base_t<MetaData> {
  // these must be private, so that they are not automatically inherited
  using value_type = Value;
  using metadata_base = metadata_base_t<MetaData>;
  using metadata_type = typename metadata_base::metadata_type;
  using options_type =
      detail::replace_default<Options, decltype(option::underflow | option::overflow)>;

  using unit_type = detail::get_unit_type<value_type>;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  constexpr unit_regular() = default;

  unit_regular(piecewise pw, metadata_type meta = {}, options_type options = {})
      : metadata_base(std::move(meta)), pw_(std::move(pw)) {
    // static_asserts were moved here from class scope to satisfy deduction in gcc>=11
    static_assert(std::is_nothrow_move_constructible<piecewise>::value,
                  "piecewise must be no-throw move constructible");
    static_assert(std::is_nothrow_move_assignable<piecewise_type>::value,
                  "piecewise must be no-throw move assignable");
    static_assert(std::is_floating_point<internal_value_type>::value,
                  "unit_regular axis requires floating point type");
    static_assert(!(options.test(option::circular) && options.test(option::growth)),
                  "circular and growth options are mutually exclusive");
    if (size() <= 0) BOOST_THROW_EXCEPTION(std::invalid_argument("bins > 0 required"));
    if (!std::isfinite(min_) || !std::isfinite(delta_))
      BOOST_THROW_EXCEPTION(
          std::invalid_argument("forward transform of start or stop invalid"));
    if (delta_ == 0)
      BOOST_THROW_EXCEPTION(std::invalid_argument("range of axis is zero"));
  }

  /// Constructor used by algorithm::reduce to shrink and rebin (not for users).
  unit_regular(const unit_regular& src, index_type begin, index_type end, unsigned merge)
      : unit_regular(src.transform(), (end - begin) / merge, src.value(begin),
                     src.value(end), src.metadata()) {
    assert(false);
    assert((end - begin) % merge == 0);
    if (options_type::test(option::circular) && !(begin == 0 && end == src.size()))
      BOOST_THROW_EXCEPTION(std::invalid_argument("cannot shrink circular axis"));
  }

  /// Return index for value argument.
  index_type index(value_type x) const noexcept { return piecewise_.index(x); }

  /// Returns index and shift (if axis has grown) for the passed argument.
  std::pair<index_type, index_type> update(value_type x) noexcept {
    assert(options_type::test(option::growth));
    return piecewise_.update(x);
  }

  /// Return value for fractional index argument.
  value_type value(real_index_type i) const noexcept {
    assert(false);
    return static_cast<value_type>(i);
  }

  /// Return bin for index argument.
  decltype(auto) bin(index_type idx) const noexcept {
    return interval_view<unit_regular>(*this, idx);
  }

  /// Returns the number of bins, without over- or underflow.
  index_type size() const noexcept { return size_; }

  /// Returns the options.
  static constexpr unsigned options() noexcept { return options_type::value; }

  // template <class V, class T, class M, class O>
  // bool operator==(const unit_regular<V, T, M, O>& o) const noexcept {
  //   return detail::relaxed_equal{}(transform(), o.transform()) && size() == o.size() &&
  //          min_ == o.min_ && delta_ == o.delta_ &&
  //          detail::relaxed_equal{}(this->metadata(), o.metadata());
  // }
  // template <class V, class T, class M, class O>
  // bool operator!=(const unit_regular<V, T, M, O>& o) const noexcept {
  //   return !operator==(o);
  // }

  // template <class Archive>
  // void serialize(Archive& ar, unsigned /* version */) {
  //   ar& make_nvp("transform", static_cast<transform_type&>(*this));
  //   ar& make_nvp("size", size_);
  //   ar& make_nvp("meta", this->metadata());
  //   ar& make_nvp("min", min_);
  //   ar& make_nvp("delta", delta_);
  // }

private:
  index_type size_{0};
  internal_value_type min_{0} piecewise piecewise_{};
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
