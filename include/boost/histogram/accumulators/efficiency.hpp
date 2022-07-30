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
#include <boost/throw_exception.hpp>
#include <cassert>
#include <limits>
#include <utility>

namespace boost {
namespace histogram {
namespace accumulators {

class deviation;
class confidence_level;

class deviation {
public:
  /// constructor from scaling factor
  explicit deviation(double d = 1) noexcept : d_{d} {
    if (d <= 0)
      BOOST_THROW_EXCEPTION(std::invalid_argument("scaling factor must be positive"));
  }

  /// implicit conversion from confidence_level
  deviation(confidence_level cl) noexcept; // need to implement confidence_level first

  /// explicit conversion to scaling factor
  explicit operator double() const noexcept { return d_; }

  friend deviation operator*(deviation d, double z) noexcept {
    return deviation(d.d_ * z);
  }
  friend deviation operator*(double z, deviation d) noexcept { return d * z; }
  friend bool operator==(deviation a, deviation b) noexcept { return a.d_ == b.d_; }
  friend bool operator!=(deviation a, deviation b) noexcept { return !(a == b); }

private:
  double d_;
};

class confidence_level {
public:
  /// constructor from confidence level (a probablity)
  explicit confidence_level(double cl) noexcept : cl_{cl} {
    if (cl <= 0 || cl >= 1)
      BOOST_THROW_EXCEPTION(std::invalid_argument("0 < cl < 1 is required"));
  }

  /// implicit conversion from deviation
  confidence_level(deviation d) noexcept
      // solve normal cdf(z) - cdf(-z) = 2 (cdf(z) - 0.5)
      : cl_{std::fma(2.0, detail::normal_cdf(static_cast<double>(d)), -1.0)} {}

  /// explicit conversion to numberical probability
  explicit operator double() const noexcept { return cl_; }

  friend bool operator==(confidence_level a, confidence_level b) noexcept {
    return a.cl_ == b.cl_;
  }
  friend bool operator!=(confidence_level a, confidence_level b) noexcept {
    return !(a == b);
  }

private:
  double cl_;
};

inline deviation::deviation(confidence_level cl) noexcept
    : d_{detail::normal_ppf(std::fma(0.5, static_cast<double>(cl), 0.5))} {}

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

  std::pair<double, double> confidence_interval(
      interval_type type = interval_type::wilson, deviation d = deviation{1.0}) const {
    const double z = static_cast<double>(d);
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

    // code should never arrive here
    assert(false);
    return std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN());
  }

  value_type n_success_ = 0;
  value_type n_failure_ = 0;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif
