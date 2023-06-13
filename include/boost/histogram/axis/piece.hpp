// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_PIECE_HPP
#define BOOST_HISTOGRAM_AXIS_PIECE_HPP

#include <algorithm>
#include <boost/core/nvp.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/axis/metadata_base.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/axis/regular.hpp> // For one_unit, get_scale_type, etc
#include <boost/histogram/detail/convert_integer.hpp>
#include <boost/histogram/detail/limits.hpp>
#include <boost/histogram/detail/relaxed_equal.hpp>
#include <boost/histogram/detail/replace_type.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11/utility.hpp>
#include <boost/throw_exception.hpp>
#include <boost/variant2/variant.hpp>
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
/** The abstract piece class that other pieces inherit from

  The class has two pure virtual functions:
    1. inverse() maps from bin space Y to input space X
    2. forward() maps from input space X to bin space Y

  This abstract class has three data members:
    1. size_ic_ number of bins
    2. x0_ lower edge of the first bin
    3. xN_ upper edge of the last bin

  The member x0_ is used internally by the forward and inverse functions. The member
  xN_ can be used by other classes to avoid calling inverse() to get the location of
  on the last bin.
*/
template <class Value, class PieceType>
class piece_base {
  using value_type = Value;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /// The mapping from bin space Y to input space X
  virtual internal_value_type inverse(internal_value_type y) const noexcept = 0;

  /// The mapping from input space X to bin space Y
  virtual internal_value_type forward(internal_value_type x) const noexcept = 0;

  /// Creates a piece for the given arguments. This function checks the orientation of
  /// the transform and makes sure that transforms are not inverted.
  template <class... Args>
  static PieceType create(Args... args) {
    auto p = PieceType(args...);
    p.xN_ = p.inverse(p.size_ic_); // Set xN_

    assert(0 < p.size_ic_);
    assert(p.x0_ < p.xN_);
    return p;
  }

  /// Calculates the width of bins. Calling calc_bin_width(0) gives the width of the first
  /// bin.
  double calc_bin_width(double i) const noexcept {
    const auto x_i_next = inverse(i + 1);
    const auto x_i = inverse(i);
    return x_i_next - x_i;
  }

  /// Calculates the width of the last bin.
  double calc_bin_width_last() const noexcept { return calc_bin_width(size_ic_ - 1); }

  /// The number of bins
  index_type size() const noexcept { return size_ic_; }

  /// The lower edge of the first bin (i.e., bin 0).
  internal_value_type x0() const noexcept { return x0_; }

  /// The upper edge of the last bin (i.e., bin N - 1).
  internal_value_type xN() const noexcept { return xN_; }

protected:
  // Constructor called by derived classes
  piece_base(int N, double x0) : size_ic_{N}, x0_{x0} {}

private:
  index_type size_ic_{-9999};
  internal_value_type x0_{-9999.0};
  internal_value_type xN_{-9999.0};
};

/** Piece for a unit transformation.

*/
template <class Value>
class piece_unit : public piece_base<Value, piece_unit<Value>> {
  using value_type = Value;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /// The mapping from bin space Y to input space X
  internal_value_type inverse(internal_value_type y) const noexcept override {
    return y + this->x0();
  }

  /// The mapping from input space X to bin space Y
  internal_value_type forward(internal_value_type x) const noexcept override {
    return x - this->x0();
  }

private:
  // Private constructor below is called by the base class.
  friend class piece_base<Value, piece_unit<Value>>;

  piece_unit(int N, double x0) : piece_base<Value, piece_unit<Value>>(N, x0) {}
};

/** Piece for a variable transformation.

*/
template <class Value>
class piece_variable : public piece_base<Value, piece_variable<Value>> {
  using value_type = Value;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /// The mapping from bin space Y to input space X
  internal_value_type inverse(internal_value_type y) const noexcept override {
    // NOTE: copied from variable.hpp
    if (y < 0) return detail::lowest<value_type>();
    if (y == this->size()) return vec_.back();
    if (y > this->size()) return detail::highest<value_type>();
    const auto k = static_cast<index_type>(y); // precond: y >= 0
    const real_index_type z = y - k;
    // check z == 0 needed to avoid returning nan when vec_[k + 1] is infinity
    return (1.0 - z) * vec_[k] + (z == 0 ? 0 : z * vec_[k + 1]);
  }

  /// The mapping from input space X to bin space Y
  internal_value_type forward(internal_value_type x) const noexcept override {
    // NOTE: copied from variable.hpp
    if (x == vec_.back()) return this->size() - 1; // TODO: support growth option
    return static_cast<index_type>(std::upper_bound(vec_.begin(), vec_.end(), x) -
                                   vec_.begin() - 1);
  }

private:
  // Private constructor below is called by the base class.
  friend class piece_base<Value, piece_variable<Value>>;

  piece_variable(const std::vector<double>& vec)
      : piece_base<Value, piece_variable<Value>>(vec.size() - 1, vec[0]), vec_{vec} {}

  std::vector<double> vec_;
};

/** Abstract variable width piece class

*/
template <class Value, class PieceType>
class piece_b0 : public piece_base<Value, PieceType> {
  using value_type = Value;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /** Finds the piece whose size is known, but bin spacing is not.

    @param n number of bins
    @param x0 starting point
    @param xN stopping point
    @param args additional arguments

    The constructor throws `std::invalid_argument` if:
      1. the bin number is zero, or
      2. x0 <= xN.
  */
  template <class... Args>
  static PieceType solve_b0(int n, double x0, double xN, Args... args) {
    // TODO: Check that x0 < xN
    const bool is_ordered = x0 < xN;
    if (!is_ordered) BOOST_THROW_EXCEPTION(std::invalid_argument("x0 must be < xN"));

    const double b0_obj = PieceType::calc_b0(n, x0, xN, args...);
    return PieceType::create(n, x0, b0_obj, args...);
  }

  /// Returns the width of the first bin.
  double b0() const noexcept { return b0_; }

  piece_b0(int N, double x0, double b0) : piece_base<Value, PieceType>(N, x0), b0_(b0) {}

private:
  friend class piece_base<Value, PieceType>;

  double b0_;
};

/** Piece for uniform spacing

   Uniform bin spacing b0

         b0     b0     b0     b0
     y=0    y=1    y=2    y=3    y=4
      |------|------|------|------|
      x0                          x4

   The sequence
     x0 = x0
     x1 = x0 + b0
     x2 = x0 + b0 * 2
     x3 = x0 + b0 * 3
     ...
     xN = x0 + b0 * yN

*/
template <class Value>
class piece_uniform : public piece_b0<Value, piece_uniform<Value>> {
  using value_type = Value;
  using unit_type = detail::get_unit_type<value_type>;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /** The mapping from bin space Y to input space X.

    Uses the formula:
       x = x0 + b0 * y
  */
  // Solves for x in terms of y
  internal_value_type inverse(internal_value_type y) const noexcept override {
    return this->x0() + y * this->b0();
  }

  /** The mapping from input space X to bin space Y.

    Solving x = x0 + b0 * y for y gives the formula:
       y = (x - x0) / b0
  */
  internal_value_type forward(internal_value_type x) const noexcept override {
    return ((x / unit_type{}) - this->x0()) / this->b0();
  }

  /** Calculates the bin width.

    Uses the formula:
       b0 = (xN - x0) / N
  */
  static double calc_b0(int N, double x0, double xN) { return (xN - x0) / N; }

  /// Calculates the next bin width. The bin width is constant.
  static double next_b(double b0) { return b0; }

private:
  // Private constructor below is called by the base class.
  friend class piece_base<Value, piece_uniform<Value>>;

  piece_uniform(int N, double x0, double b0)
      : piece_b0<Value, piece_uniform<Value>>(N, x0, b0) {}
};

/** Creates a piece where the bin size is multiplied by a constant

     b0  b0*r     b0*r²         b0*r³
   N=0 N=1   N=2         N=3                N=4
    |---|-----|-----------|------------------|
    x0                                      x4

   The sequence (for some value of ψ)
     x0 = 1  * b0 / (r - 1) + ψ
     x1 = r  * b0 / (r - 1) + ψ
     x2 = r² * b0 / (r - 1) + ψ
     x3 = r³ * b0 / (r - 1) + ψ
     ...
     xN = rᴺ * b0 / (r - 1) + ψ

   Note: the first bin spacing is b0
     x1 - x0 = r * b0 / (r - 1) - 1 * b0 / (r - 1)
             = (r - 1) * b0 / (r - 1)
             = b0

   Find a formula for N
     xN - x0  = b0 / (r - 1) * rᴺ - b0 / (r - 1) * 1
              = b0 / (r - 1) * (rᴺ - 1)

   N in terms of xN
     N = log2(((xN - x0) / b0) * (r - 1) + 1) / log2(r)

*/
template <class Value>
class piece_multiply : public piece_b0<Value, piece_multiply<Value>> {
  using value_type = Value;
  using unit_type = detail::get_unit_type<value_type>;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /** The mapping from bin space Y to input space X.

    Uses the formula:
       x = x0 + b0 * (rᴺ - 1) / (r - 1)
  */
  internal_value_type inverse(internal_value_type N) const noexcept override {
    return this->x0() + this->b0() * (std::pow(r_, N) - 1) / (r_ - 1);
  }

  /** The mapping from input space X to bin space Y.

    Solving the inverse formula for N gives the formula:
       N = log2(((x - x0) / b0) * (r - 1) + 1) / log2(r)
  */
  internal_value_type forward(internal_value_type x) const noexcept override {
    const auto z = ((x / unit_type{}) - this->x0()) / this->b0();
    return std::log2(z * (r_ - 1) + 1) / std::log2(r_);
  }

  /** Calculates the bin width for the first bin.

    Uses the formula:
       b0 = (xN - x0) * (r - 1) / (rᴺ - 1)
  */
  static double calc_b0(int N, double x0, double xN, double r) {
    return (xN - x0) * (r - 1) / (std::pow(r, N) - 1);
  }

  /// Calculates the next bin width.
  static double next_b(double b0, double r) { return b0 * r; }

  /// Returns the ratio between bin widths.
  double r() const noexcept { return r_; }

private:
  // Private constructor below is called by the base class.
  friend class piece_base<Value, piece_multiply<Value>>;

  piece_multiply(int N, double x0, double b0, double r)
      : piece_b0<Value, piece_multiply<Value>>(N, x0, b0), r_(r) {}

  double r_;
};

/** Piece for adding a constant to the bin spacing

   b0  b0+r     b0+2r         b0+3r
 N=0 N=1   N=2         N=3                N=4
  |---|-----|-----------|------------------|
  x0                                      x4

 The sequence
   x0 = x0 = 0 * b0
   x1 = x0 + 1 * b0 + r * 0
   x2 = x0 + 2 * b0 + r * 1
   x3 = x0 + 3 * b0 + r * (1 + 2)
   x4 = x0 + 4 * b0 + r * (1 + 2 + 3)
   ...
   x = x0 + N * b0 + r * (1 + 2 + ... + N-1)
   x = x0 + N * b0 + r * (N * (N - 1) / 2)

*/
template <class Value>
class piece_add : public piece_b0<Value, piece_add<Value>> {
  using value_type = Value;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /// The mapping from bin space Y to input space X
  internal_value_type inverse(internal_value_type N) const noexcept override {
    return this->x0() + N * this->b0() + r_ * (N * (N - 1)) / 2;
  }

  /** The mapping from input space X to bin space Y.

    Starting with:
      x = x0 + N * b0 + r * (N * (N - 1) / 2)
    Multiply by 2:
      2 * x = 2 * x0 + 2 * N * b0 + r * (N * (N - 1))
    Expand terms:
      2 * x = 2 * x0 + 2 * N * b0 + rN² - rN
    Collect terms by powers of N:
      0 = rN² + (2 * b0 - r)N + (2 * x0 - 2 * x)

      a = r
      b = 2 * b0 - r
      c = 2 * x0 - 2 * x
  */
  internal_value_type forward(internal_value_type x) const noexcept override {
    const double a = r_;
    const double b = 2 * this->b0() - r_;
    const double c = 2 * this->x0() - 2 * x;
    const auto roots = quadratic_roots(a, b, c);
    return roots.second; // Take the positive root
  }

  /// Finds the roots of the quadratic equation ax² + bx + c = 0.
  static std::pair<value_type, value_type> quadratic_roots(const value_type& a,
                                                           const value_type& b,
                                                           const value_type& c) noexcept {
    // https://people.csail.mit.edu/bkph/articles/Quadratics.pdf

    const value_type two_a = 2 * a;
    const value_type two_c = 2 * c;
    const value_type sqrt_bb_4ac = std::sqrt(b * b - two_a * two_c);

    if (b >= 0) {
      const value_type root1 = (-b - sqrt_bb_4ac) / two_a;
      const value_type root2 = two_c / (-b - sqrt_bb_4ac);
      return {root1, root2};
    } else {
      const value_type root1 = two_c / (-b + sqrt_bb_4ac);
      const value_type root2 = (-b + sqrt_bb_4ac) / two_a;
      return {root1, root2};
    }
  }

  /** Calculates the bin width for the first bin.

    Uses the formula:
       b0 = (xN - x0 - r * (N * (N - 1) / 2)) / N
  */
  static double calc_b0(int N, double x0, double xN, double r) {
    return (xN - x0 - r * (N * (N - 1) / 2)) / N;
  }

  /// Calculates the next bin width.
  static double next_b(double b0, double r) { return b0 + r; }

  /// Returns the ratio between bin widths.
  double r() const noexcept { return r_; }

private:
  // Private constructor below is called by the base class.
  friend class piece_base<Value, piece_add<Value>>;

  piece_add(int N, double x0, double b0, double r)
      : piece_b0<Value, piece_add<Value>>(N, x0, b0), r_(r) {}

  double r_;
};

/** Variant piece

*/
template <class Value>
class piece_variant {
  using value_type = Value;
  using internal_value_type = detail::get_scale_type<value_type>;
  using piece_variant_type =
      boost::variant2::variant<piece_unit<Value>, piece_uniform<Value>,
                               piece_multiply<Value>, piece_add<Value>,
                               piece_variable<Value>>;

public:
  template <class T>
  explicit piece_variant(const T& p) : piece_(p) {}

  internal_value_type forward(internal_value_type x) const noexcept {
    return boost::variant2::visit([x](const auto& p) { return p.forward(x); }, piece_);
  }

  internal_value_type inverse(internal_value_type x) const noexcept {
    return boost::variant2::visit([x](const auto& p) { return p.inverse(x); }, piece_);
  }

  index_type size() const noexcept {
    return boost::variant2::visit([](const auto& p) { return p.size(); }, piece_);
  }

  internal_value_type x0() const noexcept {
    return boost::variant2::visit([](const auto& p) { return p.x0(); }, piece_);
  }

  internal_value_type xN() const noexcept {
    return boost::variant2::visit([](const auto& p) { return p.xN(); }, piece_);
  }

  internal_value_type calc_bin_width(double i) const noexcept {
    return boost::variant2::visit([i](const auto& p) { return p.calc_bin_width(i); },
                                  piece_);
  }

  internal_value_type calc_bin_width_last() const noexcept {
    return boost::variant2::visit([](const auto& p) { return p.calc_bin_width_last(); },
                                  piece_);
  }

  template <class VariantType>
  bool has_variant() const noexcept {
    return boost::variant2::holds_alternative<VariantType>(piece_);
  }

  const piece_variant_type& operator()() const noexcept { return piece_; }

private:
  piece_variant_type piece_;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
