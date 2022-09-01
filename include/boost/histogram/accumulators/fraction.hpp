// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_FRACTION_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_FRACTION_HPP

#include <boost/core/nvp.hpp>
#include <boost/histogram/fwd.hpp> // for fraction<>
#include <boost/histogram/utility/wilson_interval.hpp>
#include <cassert>
#include <utility>

namespace boost {
namespace histogram {
namespace accumulators {

/** Calculates fraction of sample.

  Helps with value, variance and multiple confidence intervals (Agresti Coull, Clopper Pearson, Jeffreys, Wald and Wilson) for given sample.
*/
template <class ValueType>
class fraction {
public:
  using value_type = ValueType;
  using const_reference = const value_type&;
  using real_type = typename std::conditional<std::is_floating_point<value_type>::value,
                                              value_type, double>::type;

  fraction() noexcept = default;

  /// Initialize to external successes and failures.
  fraction(const_reference successes, const_reference failures)
      : succ_(successes), fail_(failures) {}

  /// Allow implicit conversion from fraction<T>
  template <class T>
  fraction(const fraction<T>& e) noexcept : fraction{static_cast<value_type>(e.successes()), static_cast<value_type>(e.failures())} {}

  /// Insert boolean sample x.
  void operator()(bool x) noexcept {
    if (x)
      ++succ_;
    else
      ++fail_;
  }


  /** Return successes or true accumulated boolean samples.

    successes() can simply be used to fetch the number of true
    or positive samples accumulated in the storage, and further be
    utilized to perform operations outside the given features.
  */
  value_type successes() const noexcept { return succ_; }

  /** Return failures or false accumulated boolean samples.

    failures() can simply be used to fetch the number of false
    or negative samples accumulated in the storage, and further be
    utilized to perform operations outside the given features.
  */
  value_type failures() const noexcept { return fail_; }

  /** Return total number of accumulated boolean samples.

    count() can be used to fetch all samples accumulated in the storage,
    and further be utilized to perform operations outside the given features.
  */
  value_type count() const noexcept { return succ_ + fail_; }

  /** Return success fractional value of the accumulated boolean samples.

    value() provides the hit-ratio/success-ratio/true-fraction for the
    given samples and is denoted by success samples per total samples.
  */
  real_type value() const noexcept {
    const real_type s = static_cast<real_type>(succ_);
    const real_type n = static_cast<real_type>(count());
    return s / n;
  }

  /** Return variance of the accumulated boolean samples.

    variance() provides the binomial distribution based variance
    value with number of successes as target value.
  */
  real_type variance() const noexcept {
    // We want to compute Var(p) for p = X / n with Var(X) = n p (1 - p)
    // For Var(X) see
    // https://en.wikipedia.org/wiki/Binomial_distribution#Expected_value_and_variance
    // Error propagation: Var(p) = p'(X)^2 Var(X) = p (1 - p) / n
    const real_type p = value();
    const value_type n = count();
    const real_type one{1};
    return p * (one - p) / n;
  }

  /// Return the standard interval (Wilson score interval)
  auto confidence_interval() const noexcept {
    using namespace boost::histogram::utility;

    return wilson_interval<real_type>()(successes(), failures());
  }

private:
  value_type succ_ = 0;
  value_type fail_ = 0;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif
