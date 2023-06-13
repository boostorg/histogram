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
  auto forward(T x) const noexcept {

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
