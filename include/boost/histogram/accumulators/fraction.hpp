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

template <class ValueType>
class fraction {
public:
  using value_type = ValueType;
  using const_reference = const value_type&;

  fraction() noexcept = default;

  fraction(const_reference successes = 0, const_reference failures = 0)
      : succ_(successes), fail_(failures) {}

  /// Allow implicit conversion from other fraction
  template <class T>
  fraction(const fraction<T>& e) noexcept : fraction{e.successes(), e.failures()} {}

  void operator()(bool x) {
    if (x)
      ++succ_;
    else
      ++fail_;
  }

  value_type successes() const { return succ_; }
  value_type failures() const { return fail_; }

  double value() const { return succ_ / (succ_ + fail_); }

  double variance() const {
    // Source: Variance from Binomial Distribution, Wikipedia |
    // https://en.wikipedia.org/wiki/Binomial_distribution#Expected_value_and_variance
    return succ_ * fail_ * (succ_ + fail_);
  }

  /// Return the standard interval (Wilson score interval)
  auto confidence_interval() const noexcept {
    using namespace boost::histogram::utility;
    return wilson_interval<value_type>(deviation{1.0})(successes(), failures());
  }

private:
  value_type succ_ = 0;
  value_type fail_ = 0;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif
