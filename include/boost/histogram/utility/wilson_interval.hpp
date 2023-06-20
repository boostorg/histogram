// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_WILSON_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_WILSON_INTERVAL_HPP

#include <boost/histogram/fwd.hpp>
#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <cmath>
#include <utility>

namespace boost {
namespace histogram {
namespace utility {

/**
  Wilson interval.

  The Wilson score interval is simple to compute, has good coverage. Intervals are
  automatically bounded between 0 and 1 and never empty. The interval is asymmetric.

  Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical
  inference". Journal of the American Statistical Association. 22 (158): 209-212.
  doi:10.1080/01621459.1927.10502953. JSTOR 2276774.

  The coverage probability for a random ensemble of fractions is close to the nominal
  value. Unlike the Clopper-Pearson interval, the Wilson score interval is not
  conservative. For some values of the fractions, the interval undercovers and overcovers
  for neighboring values. This is a shared property of all alternatives to the
  Clopper-Pearson interval.

  The Wilson score intervals is widely recommended for general use in the literature. For
  a review of the literature, see R. D. Cousins, K. E. Hymes, J. Tucker, Nucl. Instrum.
  Meth. A 612 (2010) 388-398.
*/
template <class ValueType>
class wilson_interval : public binomial_proportion_interval<ValueType> {
public:
  using value_type = typename wilson_interval::value_type;
  using interval_type = typename wilson_interval::interval_type;

  /** Construct Wilson interval computer.

    @param d Number of standard deviations for the interval. The default value 1
    corresponds to a confidence level of 68 %. Both `deviation` and `confidence_level`
    objects can be used to initialize the interval.
  */
  explicit wilson_interval(deviation d = deviation{1.0}) noexcept
      : z_{static_cast<value_type>(d)} {}

  using binomial_proportion_interval<ValueType>::operator();

  /** Compute interval for given number of successes and failures.

    @param successes Number of successful trials.
    @param failures Number of failed trials.
  */
  interval_type operator()(value_type successes, value_type failures) const noexcept {
    // See https://en.wikipedia.org/wiki/
    //   Binomial_proportion_confidence_interval
    //   #Wilson_score_interval

    // We make sure calculation is done in single precision if value_type is float
    // by converting all literals to value_type. Double literals in the equation
    // would turn intermediate values to double.
    const value_type half{0.5}, quarter{0.25}, zsq{z_ * z_};
    const value_type total = successes + failures;
    const value_type minv = 1 / (total + zsq);
    const value_type t1 = (successes + half * zsq) * minv;
    const value_type t2 =
        z_ * minv * std::sqrt(successes * failures / total + quarter * zsq);
    return {t1 - t2, t1 + t2};
  }

  /// Returns the third order correction for n_eff.
  static value_type third_order_correction(value_type n_eff) noexcept {
    // The approximate formula reads:
    //   f(n) = (n³ + n² + 2n + 6) / n³
    //
    // Applying the substitution x = 1 / n gives:
    //   f(n) = 1 + x + 2x² + 6x³
    //
    // Using Horner's method to evaluate this polynomial gives:
    //   f(n) = 1 + x (1 + x (2 + 6x))
    if (n_eff == 0) return 1;
    const value_type x = 1 / n_eff;
    return 1 + x * (1 + x * (2 + 6 * x));
  }

  /** Computer the confidence interval for the provided problem.

    @param p The problem to solve.
  */
  interval_type solve_for_neff_phat_correction(
      const value_type& n_eff, const value_type& p_hat,
      const value_type& correction) const noexcept {
    // Equation 41 from this paper: https://arxiv.org/abs/2110.00294
    //      (p̂ - p)² = p (1 - p) (z² f(n) / n)
    // Multiply by n to avoid floating point error when n = 0.
    //    n (p̂ - p)² = p (1 - p) z² f(n)
    // Expand.
    //    np² - 2np̂p + np̂² = pz²f(n) - p²z²f(n)
    // Collect terms of p.
    //    p²(n + z²f(n)) + p(-2np̂ - z²f(n)) + (np̂²) = 0
    //
    // This is a quadratic equation ap² + bp + c = 0 where
    //    a = n + z²f(n)
    //    b = -2np̂ - z²f(n)
    //    c = np̂²

    const value_type zz_correction = (z_ * z_) * correction;

    const value_type a = n_eff + zz_correction;
    const value_type b = -2 * n_eff * p_hat - zz_correction;
    const value_type c = n_eff * (p_hat * p_hat);

    return quadratic_roots(a, b, c);
  }

private:
  // Finds the roots of the quadratic equation ax² + bx + c = 0.
  static interval_type quadratic_roots(const value_type& a, const value_type& b,
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

  value_type z_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif