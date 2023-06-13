// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_PIECEWISE_HPP
#define BOOST_HISTOGRAM_AXIS_PIECEWISE_HPP

#include <algorithm>
#include <boost/core/nvp.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/axis/metadata_base.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/axis/piece.hpp>
#include <boost/histogram/axis/regular.hpp>
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

/**

Solution overview:

This 1D piecewise axis implementation has two layers. The first layer has to
do with creating immutable transformations from input space X to bin space Y.
The second layer modifies the behavior of these transformation in order to make
them behave like an axis. If axis growth is allowed, this second layer is mutable.
I explain how this works on a high level.

Layer 1: Transformation
Transformation pieces:

Suppose we want to divide the input space X into 4 bins of unit spacing starting
at x=3.

bin   0   1   2   3
    |---|---|---|---|
   x=3             x=7

The transformation is described by three pieces of information
  forward transformation: y = f(x) = x - 3
  inverse transformation: x = f⁻¹(y) = y + 3
  number of bins: N = 4

There are five supported transformation types:
  1. unit      (bin spacing is 1)
  2. uniform   (bin spacing is constant)
  3. multiply  (bin spacing is multiplied by a constant)
  4. add       (bin spacing is added to by a constant)
  5. arbitrary (bin bounds are user supplied points)


Combining transformations:

Suppose we wanted to combine the following two transformations.

  bin   0   1   2   3          bin  0  1  2   3    4
      |---|---|---|---|            |-|--|---|----|-----|
     x=3             x=7          x=9                x=15
         Unit spacing                   Add spacing

Typically one wouldn't want to combine two transformations with a gap between them,
but this possibility is supported by the implementation. A piecewise transformation
stores each of these unmodified transformations internally. The external behavior
is that of transformation with bins 0 to 8 shown below.

  bin   0   1   2   3     nan    4  5  6   7    8
      |---|---|---|---|         |-|--|---|----|-----|
     x=3             x=7       x=9                x=15


Layer 2: Axis behavior
Bin shifting:

Suppose we wanted to use the transformation above in a growable axis. This axis starts
off with 9 bins. Two kinds of bins can be added: underflow and overflow. Underflow bins
are added below x=3. Overflow bins are added above x=15. Adding bins between x=7 and
x=9 is not supported. The mutable transformation `x_bin_shift` accumulates:
  - added underflow bins, and
  - added overflow bins.
Using this information, x_bin_shift augments the behavior of the contained piecewise
transformation as follows. The size (i.e., number of bins) of the transformation is
increased by the number of underflow and overflow bins. The bin values (i.e., Y axis)
is shifted by the number of underflow bins in the forward and inverse calculations.

Linear axis:

Wrapping the, possibly shifted, transformation in an `int_resolver_linear` makes the
transformation behave like an axis. This linear int resolver implemenets an index method.

Circular axis:

Suppose we wanted to make the above described piecewise transformation circular with
a relevant period x=3 to x=23. This transformaion is copied below for convenience.

  bin   0   1   2   3     nan    4  5  6   7    8
      |---|---|---|---|         |-|--|---|----|-----|
     x=3             x=7       x=9                x=15

To do this, we wrap the transformation in an `int_resolver_circular` with the bounds
x=3 and x=23. This circular transform augments the behavior of the contained
transformation. The input (i.e., the X axis) is wrapped to the domain x=3 to x=23
before being given to the contained transformation. This circular int resolver
implemenets an index method allowing it to act as an axis.

*/

/** Piecewise -- putting pieces together

*/
template <class Value, class PieceType>
class piecewise {
  using value_type = Value;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /// Creates a piecewise starting with an initial piece.
  template <class T>
  explicit piecewise(const T& p) {
    v_pieces_.push_back(PieceType(p));
  }

  /// The mapping from input space X to bin space Y
  template <class T>
  T forward(T x) const noexcept {

    // First piece
    const auto& piece_0 = v_pieces_.front();
    if (x < piece_0.xN()) return piece_0.forward(x);

    // Middle pieces
    index_type offset = piece_0.size();
    const int n_middle = v_pieces_.size() - 1;
    for (int j = 1; j < n_middle; ++j) {
      const auto& p = v_pieces_[j];

      if (x < p.x0()) {
        return std::numeric_limits<T>::quiet_NaN(); // Between discontinuous pieces
      } else if (x < p.xN()) {
        return offset + p.forward(x);
      }

      offset += p.size();
    }

    const auto& piece_N = get_piece_R();
    if (piece_N.x0() <= x) {
      return offset + get_piece_R().forward(x); // Last piece
    } else {
      return std::numeric_limits<T>::quiet_NaN(); // Before last piece
    }
  }

  /** Extrapolates the bin spacing of the right side of the piecewise axis.

    @param N number of bins to extrapolate
    @param args arguments to pass to the piece constructor

    Example: you have a piecewise axis with unit spacing and you want to double the
    spacing of the right side for 3 bins. The following code will do this.

      this->extrapolate_R<piece_multiply>(3, 2.0);

                 <- - - -new piece- - - - ->
      |-|-|-|-|-|---|-------|---------------|
      3 4 5 6 7 8 9 10      14              22
  */
  template <class P, class... Args>
  void extrapolate_R(int N, Args... args) {
    double x0 = xN(); // Right side of the right piece becomes x0 for the new piece
    double b_prev = get_piece_R().calc_bin_width_last();
    double b0 = P::next_b(b_prev, args...);
    const auto p = P::create(N, x0, b0, args...);
    add_R(p);
  }

  /** Adds a piece to the right side of the piecewise axis.

    @param p piece to add

    The constructor throws `std::invalid_argument` if the left side of the new piece
    does not exactly match the right side of the right piece.
  */
  template <class P>
  void add_R(const P& p) {
    if (p.x0() != xN()) // Right side of the right piece is not equal to the left side of
                        // the new piece
      BOOST_THROW_EXCEPTION(std::invalid_argument("x0 != xN"));
    add_R_gap_okay(p);
  }

  /** Adds a piece to the right side of the piecewise axis.

    @param p piece to add

    The constructor throws `std::invalid_argument` if the left side of the new piece
    is less than the right side of the right piece.
  */
  template <class P>
  void add_R_gap_okay(const P& p) {
    // Check that the left side of the new piece is equal to or greater than the right
    // side of the right piece
    if (p.x0() < xN()) BOOST_THROW_EXCEPTION(std::invalid_argument("x0 < xN"));

    v_pieces_.push_back(PieceType(p));
  }

  /// The number of bins in the piecewise.
  index_type size() const noexcept {
    index_type size = 0;
    for (const auto& p : v_pieces_) { size += p.size(); }
    return size;
  }

  /// The location of the right side of the right piece.
  double xN() const noexcept { return get_piece_R().xN(); }

  /// The location of the left side of the left piece.
  double x0() const noexcept { return v_pieces_.front().x0(); }

private:
  // Gets the right-most piece.
  const PieceType& get_piece_R() const noexcept {
    assert(!v_pieces_.empty());
    return v_pieces_.back();
  }

  std::vector<PieceType> v_pieces_{};
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
