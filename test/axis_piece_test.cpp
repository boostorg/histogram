// CopyR 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/axis/piece.hpp>
#include <limits>
#include <sstream>
#include <type_traits>
#include "axis.hpp"
#include "is_close.hpp"
#include "std_ostream.hpp"
#include "str.hpp"
#include "throw_exception.hpp"

#include <cmath>
#include <numeric>

using namespace boost::histogram;

//
class tester_piece_b0 {
public:
  static tester_piece_b0 case_1() { return tester_piece_b0(4, -1.1, 1.2); }

  tester_piece_b0(int bins, double x0, double b0) : size_(bins), x0_(x0), b0_(b0) {}

  template <class PieceType, class... Args>
  PieceType create(Args... args) const {
    return PieceType::create(size_, x0_, b0_, args...);
  }

  template <class PieceType, typename Lambda>
  void test_create(const PieceType& p, Lambda&& lambda) const {

    // Check abstract piece member data
    BOOST_TEST_EQ(p.size(), size_);
    BOOST_TEST_EQ(p.x0(), x0_);
    // xN is tested below

    double bin_width = b0_;
    double x = x0_;
    for (int i = 0; i < size_; ++i) {
      // Check that forward forward transformation (x --> float_index) is correct
      BOOST_TEST_IS_CLOSE(p.forward(x), i, 1.0e-12);
      x += bin_width;

      // Check that the bin width is correct
      BOOST_TEST_IS_CLOSE(p.calc_bin_width(i), bin_width, 1.0e-12);
      bin_width = lambda(bin_width);
    }

    // Check xN
    BOOST_TEST_IS_CLOSE(p.xN(), x, 1.0e-12);

    // Check that inverse transformation is correct
    for (double v = -9.0; v < 9.0; v += 0.01) {
      BOOST_TEST_IS_CLOSE(p.inverse(p.forward(v)), v, 1.0e-12);
    }
  }

  int size() const { return size_; }
  double x0() const { return x0_; }
  double b0() const { return b0_; }

private:
  int size_;
  double x0_;
  double b0_;
};

// Tests that two pieces are equal
template <class PieceType>
void test_equal(const PieceType& p1, const PieceType& p2) {
  BOOST_TEST_EQ(p1.size(), p2.size());
  BOOST_TEST_IS_CLOSE(p1.x0(), p2.x0(), 1.0e-12);
  BOOST_TEST_IS_CLOSE(p1.xN(), p2.xN(), 1.0e-12);

  // Checks (indirectly) that the parameters of the piece are the same
  for (int i = 0; i < p1.size(); ++i) {
    BOOST_TEST_IS_CLOSE(p1.calc_bin_width(i), p2.calc_bin_width(i), 1.0e-12);
  }
}

template <class PieceType, class... Args>
void test_solve_b0(const PieceType& p_R, Args... args) {
  // Reconstruct piece with xN instead of b0
  const auto p2 = PieceType::solve_b0(p_R.size(), p_R.x0(), p_R.xN(), args...);

  // See if pieces are equal
  test_equal(p_R, p2);
}

int main() {
  BOOST_TEST(std::is_nothrow_move_assignable<axis::piece_uniform<double>>::value);
  BOOST_TEST(std::is_nothrow_move_constructible<axis::piece_uniform<double>>::value);

  // Test unit
  {
    using p_type = axis::piece_unit<double>;
    const auto p = p_type::create(4, 2);

    BOOST_TEST_EQ(p.size(), 4);
    BOOST_TEST_EQ(p.x0(), 2);
    BOOST_TEST_EQ(p.xN(), 6);

    BOOST_TEST_EQ(p.forward(2), 0);
    BOOST_TEST_EQ(p.forward(3), 1);
    BOOST_TEST_EQ(p.forward(4), 2);
    BOOST_TEST_EQ(p.forward(5), 3);
    BOOST_TEST_EQ(p.forward(6), 4);

    BOOST_TEST_EQ(p.inverse(0), 2);
    BOOST_TEST_EQ(p.inverse(1), 3);
    BOOST_TEST_EQ(p.inverse(2), 4);
    BOOST_TEST_EQ(p.inverse(3), 5);
    BOOST_TEST_EQ(p.inverse(4), 6);
  }

  // Test variable
  {
    using p_type = axis::piece_variable<double>;
    const auto p = p_type::create(std::vector<double>{1.1, 2.6, 3.8, 4.2, 5.9});

    BOOST_TEST_EQ(p.size(), 4);
    BOOST_TEST_EQ(p.x0(), 1.1);
    BOOST_TEST_EQ(p.xN(), 5.9);

    BOOST_TEST_EQ(p.forward(1.1), 0);
    BOOST_TEST_EQ(p.forward(2.6), 1);
    BOOST_TEST_EQ(p.forward(3.8), 2);
    BOOST_TEST_EQ(p.forward(4.2), 3);
    // BOOST_TEST_EQ(p.forward(5.9), 4);  // Test broken

    BOOST_TEST_EQ(p.inverse(0), 1.1);
    BOOST_TEST_EQ(p.inverse(1), 2.6);
    BOOST_TEST_EQ(p.inverse(2), 3.8);
    BOOST_TEST_EQ(p.inverse(3), 4.2);
    BOOST_TEST_EQ(p.inverse(4), 5.9);
  }

  // Test b0 pieces
  {
    const tester_piece_b0 p1 = tester_piece_b0::case_1();

    // Piece uniform
    {
      using p_type = axis::piece_uniform<double>;
      const auto p = p1.create<p_type>();
      auto lambda_b_next = [](double b) { return b; };
      {
        p1.test_create(p, lambda_b_next);
        test_solve_b0(p);
      }

      // Variant
      {
        const auto p_orig = p1.create<p_type>();
        const auto p = axis::piece_variant<double>(p_orig);
        BOOST_TEST(p.has_variant<axis::piece_uniform<double>>());
        BOOST_TEST(!p.has_variant<axis::piece_multiply<double>>());
        p1.test_create(p, lambda_b_next);
      }
    }

    // Piece multiply
    {
      using p_type = axis::piece_multiply<double>;
      const double k_r_multiply = 1.1;
      auto lambda_b_next = [&](double b) { return b * k_r_multiply; };
      {
        const auto p = p1.create<p_type>(k_r_multiply);
        p1.test_create(p, lambda_b_next);
        test_solve_b0(p, k_r_multiply);
      }

      // Variant
      {
        const auto p_orig = p1.create<p_type>(k_r_multiply);
        const auto p = axis::piece_variant<double>(p_orig);
        BOOST_TEST(p.has_variant<axis::piece_multiply<double>>());
        BOOST_TEST(!p.has_variant<axis::piece_uniform<double>>());
        p1.test_create(p, lambda_b_next);
      }
    };

    // Piece add
    {
      using p_type = axis::piece_add<double>;
      const double k_r_add = 0.05;
      auto lambda_b_next = [&](double b) { return b + k_r_add; };
      {
        const auto p = p1.create<p_type>(k_r_add);
        p1.test_create(p, lambda_b_next);
        test_solve_b0(p, k_r_add);
      }

      // Variant
      {
        const auto p_orig = p1.create<p_type>(k_r_add);
        const auto p = axis::piece_variant<double>(p_orig);
        BOOST_TEST(p.has_variant<axis::piece_add<double>>());
        BOOST_TEST(!p.has_variant<axis::piece_uniform<double>>());
        p1.test_create(p, lambda_b_next);
      }
    }
  }

  return boost::report_errors();
}
