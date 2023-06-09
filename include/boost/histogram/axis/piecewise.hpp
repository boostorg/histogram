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

#include <iostream>

namespace boost {
namespace histogram {

namespace axis {

/// Class for starting point
class x_start {
public:
  x_start(double value) : value_{value} {
    if (!std::isfinite(value))
      BOOST_THROW_EXCEPTION(std::invalid_argument("x_start must be finite"));
  }

  double value() const { return value_; }

private:
  double value_;
};

/// Class for stopping point
class x_stop {
public:
  x_stop(double value) : value_{value} {
    if (!std::isfinite(value))
      BOOST_THROW_EXCEPTION(std::invalid_argument("x_stop must be finite"));
  }

  double value() const { return value_; }

private:
  double value_;
};

/// Class for width_start
class width_start {
public:
  width_start(double value) : value_{value} {
    if (!(0 < value))
      BOOST_THROW_EXCEPTION(std::invalid_argument("width_start must be > 0"));
  }

  double value() const { return value_; }

private:
  double value_;
};

/** Class that stores the bin spacing option and constant value

 There are three options for varying bin sizes.

      Uniform                   |------|------|------|------|
       (bi+1 = bi = constant)            bi     bi+1

      Multiply                  |---|-------|---------------|
       (bi+1 = bi * constant)         bi      bi+1

      Add                       |---|-----|-------|---------|
       (bi+1 = bi + constant)         bi    bi+1
*/
class bin_trans {
public:
  enum class enum_space { e_uniform, e_multiply, e_add };

  static bin_trans uniform(double value) {
    return bin_trans(enum_space::e_uniform, value);
  }
  static bin_trans multiply(double value) {
    return bin_trans(enum_space::e_multiply, value);
  }
  static bin_trans add(double value) { return bin_trans(enum_space::e_add, value); }

  bin_trans mirror() const noexcept {
    if (is_uniform()) return uniform(value_);
    if (is_multiply()) return multiply(1 / value_);
    if (is_add()) return add(-value_);
    assert(false); // TODO: make unnecessary
  }

  bool is_uniform() const { return spacing_ == enum_space::e_uniform; }
  bool is_multiply() const { return spacing_ == enum_space::e_multiply; }
  bool is_add() const { return spacing_ == enum_space::e_add; }

  enum_space spacing() const { return spacing_; }
  double value() const { return value_; }

private:
  bin_trans(enum_space spacing, double value) : spacing_{spacing}, value_{value} {}

  enum_space spacing_{enum_space::e_uniform};
  double value_;
};

/** Class for a piece of a piecewise axis

  Each piece of a piecewise axis has a start point x0 and an end point xN.
  It is required that x0 < xN.

    |--------------------------------------|
    x0                                     xN

  Each piece is divided into n bins. The bin spacing is determined by the bin spacing
  option and value.

  There are two ways to create a piece: rollout and force. With the rollout method, bins
  are added one at a time. If the bin number is positive (n > 0), n bins are added to the
  right. If the bin number is negative (n < 0), |n| bins are added to the left. With the
  force method, one gives the interval x0 to xN and the number of bins n. The problem
  finds the bin width.

*/
template <class Value>
class piece {
  using value_type = Value;
  using unit_type = detail::get_unit_type<value_type>;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /** Creates a piece with the rollout method

    @param n number of bins (right if positive, left if negative)
    @param x_start starting point
    @param width_start starting bin width
    @param bt bin spacing option and value

    The constructor throws `std::invalid_argument` if the bin number is zero or if the
    bin spacing option is invalid.

    Example: Start at x = 1 with an initial bin size of 1. Rollout 4 bins to the right,
             doubling the bin size each time.
     |-|---|-------|---------------|
   0   2   4   6   8   10  12  14  16

  */
  static piece<Value> rollout(int n, x_start x, width_start b0, bin_trans bt) {
    assert(n != 0);
    if (0 < n) {
      return rollout_right(n, x, b0, bt);
    } else {
      return rollout_left(n, x, b0, bt);
    }
  }

  /** Creates a piece with the force method

     @param n number of bins
     @param x_start starting point
     @param x_stop stopping point
     @param bt bin spacing option and value

     The constructor throws `std::invalid_argument` if:
       1. the bin number is zero,
       2. x_stop <= x_start, or
       3. the bin spacing option is invalid.

     Example: Input interval [1, 16]

       |-----------------------------|
     0   2   4   6   8   10  12  14  16

     Place 4 bins. Each bin is double the size of the previous bin.

      |-|---|-------|---------------|
    0   2   4   6   8   10  12  14  16
  */
  static piece<Value> force(int n, x_start x_start, x_stop x_stop, bin_trans bt) {
    // TODO: Check that x_start < x_stop
    const bool is_ordered = x_start.value() < x_stop.value();
    if (!is_ordered)
      BOOST_THROW_EXCEPTION(std::invalid_argument("x_start must be < x_stop"));

    const width_start b0_obj = piece::calc_b0(x_start, x_stop, n, bt);
    return piece(n, x_start, b0_obj, bt);
  }

  template <class T>
  T forward(T x) const noexcept {
    static_assert(std::is_floating_point<T>::value, "T must be a floating point type");
    // Runs in hot loop, please measure impact of changes
    T f_x;
    if (bt_.is_uniform()) {
      f_x = forward_uniform(x);
    } else if (bt_.is_multiply()) {
      f_x = forward_multiply(x);
    } else if (bt_.is_add()) {
      f_x = forward_add(x);
    } else {
      assert(false); // TODO: make unnecessary
    }
    return f_x + accumulated_left_shift_;
  }

  index_type size() const noexcept {
    return size_ic_ + accumulated_left_shift_ + accumulated_right_shift_;
  }

  double x0() const noexcept { return x0_; }
  double b0() const noexcept { return b0_; }
  bin_trans bt() const noexcept { return bt_; }
  index_type size_ic() const noexcept { return size_ic_; }

  internal_value_type xN() const noexcept { return xN_; }

  index_type accumulated_left_shift() const noexcept { return accumulated_left_shift_; }
  index_type accumulated_right_shift() const noexcept { return accumulated_right_shift_; }

  // Bin ending at x_i has width
  double calc_bin_width(double i) const noexcept {
    const auto x_i = reverse(i);
    const auto x_i_prev = reverse(i - 1);
    return x_i - x_i_prev;
  }

  static width_start calc_b0(x_start x0, x_stop xN, int N, bin_trans bt) {
    double b0;
    if (bt.is_uniform()) {
      b0 = calc_b0_uniform(x0, xN, N);
    } else if (bt.is_multiply()) {
      b0 = calc_b0_multiply(x0, xN, N, bt);
    } else if (bt.is_add()) {
      b0 = calc_b0_add(x0, xN, N, bt);
    } else {
      assert(false); // TODO: make unnecessary
    }
    return width_start(b0);
  }

  template <class T>
  internal_value_type reverse(T x) const noexcept {
    if (bt_.is_uniform()) {
      return reverse_uniform(x);
    } else if (bt_.is_multiply()) {
      return reverse_multiply(x);
    } else if (bt_.is_add()) {
      return reverse_add(x);
    } else {
      assert(false); // TODO: make unnecessary
    }
  }

private:
  static piece<Value> rollout_right(int n, x_start xL, width_start b0, bin_trans bt) {
    assert(0 < n);
    return piece(n, xL, b0, bt);
  }

  static piece<Value> rollout_left(int n_neg, x_start xR, width_start b0_R,
                                   bin_trans bt_right) {
    assert(n_neg < 0);
    const int n_pos = -n_neg;

    // Roll out right
    const piece piece_right(n_pos, xR, b0_R, bt_right);

    // Get width of last bin of right piece
    const double width_last_bin = piece_right.calc_bin_width(n_pos);

    // Get location of last bin of right piece
    const double x_last_bin = piece_right.reverse(n_pos);

    // Reflect last bin of right piece to get xL
    const double delta = x_last_bin - xR.value();
    const double xL = xR.value() - delta;

    return piece(n_pos, x_start(xL), width_start(width_last_bin), bt_right.mirror());
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
  //   xN = x0 + b0 * N
  //
  // Formula for N in terms of xN
  //   N = (xN - x0) / b0
  //
  template <class T>
  internal_value_type forward_uniform(T x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    return ((x / unit_type{}) - x0_) / b0_;
  }

  // Starting with:
  //   xN = x0 + b0 * N
  // Solve for b0 to get:
  //   b0 = (xN - x0) / N
  static double calc_b0_uniform(x_start x0, x_stop xN, int N) {
    return (xN.value() - x0.value()) / N;
  }

  template <class T>
  internal_value_type reverse_uniform(T x) const noexcept {
    return x0_ + x * b0_;
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
  //   xN = rᴺ * b0 / (r - 1) + ψ
  //
  // Note: the first bin spacing is b0
  //   x1 - x0 = r * b0 / (r - 1) - 1 * b0 / (r - 1)
  //           = (r - 1) * b0 / (r - 1)
  //           = b0
  //
  // Find a formula for N
  //   xN - x0  = b0 / (r - 1) * rᴺ - b0 / (r - 1) * 1
  //            = b0 / (r - 1) * (rᴺ - 1)
  //
  // N in terms of xN
  //   N = log2(((xN - x0) / b0) * (r - 1) + 1) / log2(r)
  //
  template <class T>
  internal_value_type forward_multiply(T x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    const auto z = forward_uniform(x); // z = (x - x0) / b0
    const auto r = bt_.value();
    return std::log2(z * (r - 1) + 1) / std::log2(r);
  }

  // Starting with:
  //    xN - x0 = b0 / (r - 1) * (rᴺ - 1)
  // Solve for b0 to get:
  //    b0 = (xN - x0) * (r - 1) / (rᴺ - 1)
  static double calc_b0_multiply(x_start x0, x_stop xN, int N, bin_trans bt) {
    const auto r = bt.value();
    return (xN.value() - x0.value()) * (r - 1) / (std::pow(r, N) - 1);
  }

  template <class T>
  internal_value_type reverse_multiply(T x) const noexcept {
    const auto r = bt_.value();
    return x0_ + b0_ * (std::pow(r, x) - 1) / (r - 1);
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
  // Multiply by 2
  //    2 * x = 2 * x0 + 2 * N * b0 + r * (N * (N - 1))
  // Expand terms:
  //    2 * x = 2 * x0 + 2 * N * b0 + rN² - rN
  // Put all terms on one side:
  //   0 = 2 * x0 + 2 * N * b0 + rN² - rN - 2 * x
  // Solve quadratic equation and take positive root
  //  N = (-1 + sqrt(1 + 8 * (x - x0) / b0 * r)) / 2
  //
  template <class T>
  internal_value_type forward_add(T x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    const auto z = forward_uniform(x); // z = (x - x0) / b0
    const auto r = bt_.value();
    return (-1 + std::sqrt(1 + 8 * z * r)) / 2;
  }

  // Starting with:
  //    2 * x = 2 * x0 + 2 * N * b0 + r * (N * (N - 1))
  // Solve for b0 to get:
  //   b0 = 2 * (xN - x0) / (N * (N - 1)) - r
  //
  static double calc_b0_add(x_start x0, x_stop xN, int N, bin_trans bt) {
    const auto r = bt.value();
    return 2 * (xN.value() - x0.value()) / (N * (N - 1)) - r;
  }

  template <class T>
  internal_value_type reverse_add(T x) const noexcept {
    const auto r = bt_.value();
    return x0_ + x * b0_ + r * (x * (x - 1)) / 2;
  }

  // Private constructor
  piece(int n, x_start x0, width_start b0, bin_trans bt)
      : size_ic_(n), x0_(x0.value()), b0_(b0.value()), bt_(bt) {
    xN_ = reverse(size_ic_);
  }

  index_type size_ic_{-9999};
  internal_value_type x0_{-9999.0};
  double b0_{-9999.0};
  bin_trans bt_{};

  internal_value_type xN_{-9999.0};

  index_type accumulated_left_shift_{0};
  index_type accumulated_right_shift_{0};
};

/***Extrapolation and Attachment***

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
            P::x_start(1),               // Start at x_start = 1
            P::width_start(1),           // Start with a bin size of width_start = 1
            P::bin_trans::uniform(1),    // Add bins of constant size 1
            P::stop::n_bins(4));         // Add 4 bins

pa.add_right(                          // Add to the right side
             P::bin_trans::multiply(2),  // Extrapolate, doubling the bin size each time
             P::stop::x(16));            // Stop when the piece contains x = 16

pa.add_left(                           // Add to the left side
            P::width_start(1),           // Start with a bin size of width_start = 1
            P::bin_trans::add(1),        // Add bins of constant size 1
            P::stop::x(4));              // Stop when the piece contains x = 4
*/
template <class Value>
class piecewise {
  using value_type = Value;
  using unit_type = detail::get_unit_type<value_type>;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /// Construct a piecewise axis with an initial piece.
  explicit piecewise(piece<value_type> p) { v_pieces_.push_back(p); }

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

  /// Shifts the axis
  void shift_axis(index_type n) {
    if (n < 0) {
      v_pieces_.front().shift_axis(n);
    } else if (0 < n) {
      // Get last element of v_pieces
      v_pieces_.back().shift_axis(n);
    }
  }

  void add_right(int n, bin_trans bt) {
    double width_start = get_right_width_start();
    add_right(n, width_start, bt);
  }
  void add_right(int n, width_start width_start, bin_trans bt) {
    double x_start = get_right_x();

    const auto p = piece<value_type>::rollout(n, x_start, width_start, bt);
    v_pieces_.push_back(p);
  }

  index_type size() const noexcept {
    index_type size = 0;
    for (const auto& p : v_pieces_) { size += p.size(); }
    return size;
    // One liner for the above function:
    // return std::accumulate(v_pieces_.begin(), v_pieces_.end(), 0);
  }

private:
  const piece<value_type>& get_left() const noexcept {
    assert(!v_pieces_.empty());
    return v_pieces_.front();
  }

  const piece<value_type>& get_right() const noexcept {
    assert(!v_pieces_.empty());
    return v_pieces_.back();
  }

  double get_left_x() const noexcept { return get_left().x0; }
  double get_right_x() const noexcept { return get_right().x1; }

  double get_left_width_start() const noexcept { return get_left().width_start; }
  double get_right_width_start() const noexcept { return get_right().width_start; }

  std::vector<piece<value_type>> v_pieces_{};
};

///
template <class Value, class Payload>
class int_shifter {
  using value_type = Value;

public:
  /// Shifts the axis
  void shift_axis(index_type n) {
    if (n < 0) {
      // Need to offset by this in the future
      accumulated_left_shift_ += std::abs(n);
    } else if (0 < n) {
      // Need to offset by this in the future
      accumulated_right_shift_ += n;
    }
  }

private:
  Payload payload_;
  index_type accumulated_left_shift_{0};
  index_type accumulated_right_shift_{0};
};

///
template <class Value, class Payload>
class int_resolver {
  using value_type = Value;

public:
  index_type index(value_type x) const noexcept {
    // Runs in hot loop, please measure impact of changes

    y = payload_.index(x);

    if (y < size()) {
      if (0 <= y)
        return static_cast<index_type>(y); // 0 <= i < size
      else
        return -1; // i < 0
    }

    // upper edge of last bin is inclusive if overflow bin is not present
    if constexpr (std::is_floating_point<value_type>::value) {
      if (!options_type::test(option::overflow) && y == size()) return size() - 1;
    }

    return size(); // also returned if x is NaN
  }

  std::pair<index_type, index_type> update(value_type x) noexcept {
    // Runs in hot loop, please measure impact of changes

    y = payload_.index(x);

    if (y < size()) {
      if (0 <= y) {
        const auto i_int = static_cast<axis::index_type>(y);
        return {i_int, 0};
      } else if (y != -std::numeric_limits<internal_value_type>::infinity()) {
        const auto i_int = static_cast<axis::index_type>(std::floor(y));
        shift_axis(i_int);
        return {0, -i_int};
      } else {
        // i is -infinity
        return {-1, 0};
      }
    }
    // i either beyond range, infinite, or NaN
    if (y < std::numeric_limits<internal_value_type>::infinity()) {
      const auto i_int = static_cast<axis::index_type>(y);
      const auto n = i_int - size() + 1;
      shift_axis(n);
      return {y, -n};
    }
    // z either infinite or NaN
    return {size(), 0};
  }

  index_type size() const noexcept { return payload_.size(); }

private:
  Payload payload_;
};

///
template <class Value, class Payload>
class int_resolver_circular {
  using value_type = Value;

public:
  index_type index(value_type x) const noexcept {
    // Runs in hot loop, please measure impact of changes

    y = payload_.index(x);

    // Not finite --> overflow bin
    if constexpr (std::is_floating_point<value_type>::value) {
      if (!std::isfinite(y)) return payload_.size();
    }

    return y % payload_.size();
  }

private:
  Payload payload_;
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
class unit_regular : public iterator_mixin<unit_regular<Value, MetaData, Options>>,
                     // protected detail::replace_default<Transform, transform::id>,
                     public metadata_base_t<MetaData> {
  // these must be private, so that they are not automatically inherited
  using value_type = Value;
  using metadata_base = metadata_base_t<MetaData>;
  using metadata_type = typename metadata_base::metadata_type;
  using options_type =
      detail::replace_default<Options, decltype(option::underflow | option::overflow)>;

  using unit_type = detail::get_unit_type<value_type>;
  using internal_value_type = detail::get_scale_type<value_type>;
  using piecewise_type = piecewise<value_type>;

public:
  constexpr unit_regular() = default;

  unit_regular(piecewise<value_type> pw, metadata_type meta = {},
               options_type options = {})
      : metadata_base(std::move(meta)), piecewise_(std::move(pw)) {
    // static_asserts were moved here from class scope to satisfy deduction in gcc>=11
    static_assert(std::is_nothrow_move_constructible<piecewise_type>::value,
                  "piecewise must be no-throw move constructible");
    static_assert(std::is_nothrow_move_assignable<piecewise_type>::value,
                  "piecewise must be no-throw move assignable");
    static_assert(std::is_floating_point<internal_value_type>::value,
                  "unit_regular axis requires floating point type");
    static_assert(!(options.test(option::circular) && options.test(option::growth)),
                  "circular and growth options are mutually exclusive");
    if (size() <= 0) BOOST_THROW_EXCEPTION(std::invalid_argument("bins > 0 required"));
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
  internal_value_type min_{0};
  piecewise<value_type> piecewise_{};
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
