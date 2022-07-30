// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_EFFICIENCY_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_EFFICIENCY_HPP

#include <boost/core/nvp.hpp>
#include <boost/histogram/detail/normal.hpp>
#include <boost/histogram/detail/wald_interval.hpp>
#include <boost/histogram/detail/wilson_interval.hpp>
#include <boost/histogram/fwd.hpp> // for efficiency<>
#include <limits>
#include <utility>

namespace boost {
namespace histogram {
namespace accumulators {

template <class ValueType>
class efficiency {

public:
  using value_type = ValueType;
  using const_reference = const value_type&;

  enum class interval_type { wald, wilson, jeffreys, agresti_coull, clopper_pearson };

  efficiency() noexcept = default;

  efficiency(const_reference n_success = 0, const_reference n_failure = 0)
      : n_success_(n_success), n_failure_(n_failure) {}

  /// Allow implicit conversion from other efficiency
  template <class T>
  efficiency(const efficiency<T>& e) noexcept : efficiency{e.successes(), e.failures()} {}

  void operator()(bool x) {
    if (x)
      ++n_success_;
    else
      ++n_failure_;
  }

  value_type successes() const { return n_success_; }
  value_type failures() const { return n_failure_; }

  value_type value() const { return n_success_ / (n_success_ + n_failure_); }

  value_type variance() const {
    // Source: Variance from Binomial Distribution, Wikipedia |
    // https://en.wikipedia.org/wiki/Binomial_distribution#Expected_value_and_variance
    return n_success_ * n_failure_ * (n_success_ + n_failure_);
  }

  std::pair<double, double> confidence_interval(interval_type type = interval_type::wald,
                                                double cl = 0.683) const {
    // solve normal cdf(z) - cdf(-z) = cl; since pdf is symmetric around 0, we have
    // cdf(z) = cl/2 - 0.5 -> z = ppf(cl / 2 - 0.5)
    double z = detail::normal_ppf(std::fma(0.5, cl, -0.5));
    double p = n_success_ / (n_success_ + n_failure_);
    switch (type) {
      case interval_type::wald: {
        double interval = detail::wald_interval(n_failure_, n_success_, z);
        return std::make_pair((p - interval), (p + interval));
      }
      case interval_type::wilson: {
        double interval = detail::wilson_interval(n_failure_, n_success_, z);
        return std::make_pair((p - interval), (p + interval));
      }
      case interval_type::jeffreys: return std::make_pair(0, 0); // implement if needed
      case interval_type::clopper_pearson:
        return std::make_pair(0, 0); // implement if needed
      case interval_type::agresti_coull:
        return std::make_pair(0, 0); // implement if needed
    };
    return std::make_pair(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()); // code should never arrive here
  }

  value_type n_success_ = 0;
  value_type n_failure_ = 0;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif
