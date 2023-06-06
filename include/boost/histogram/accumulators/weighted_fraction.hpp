// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_FRACTION_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_FRACTION_HPP

#include <boost/core/nvp.hpp>
#include <boost/histogram/accumulators/fraction.hpp>
#include <boost/histogram/detail/square.hpp>
#include <boost/histogram/fwd.hpp> // for weighted_fraction<>
#include <boost/histogram/weight.hpp>
#include <type_traits> // for std::common_type

namespace boost {
namespace histogram {
namespace accumulators {

namespace internal {

// Accumulates the sum of weights squared.
template <class ValueType>
class sum_of_weights_squared {
public:
  using value_type = ValueType;
  using const_reference = const value_type&;

  sum_of_weights_squared() = default;

  // Allow implicit conversion from sum_of_weights_squared<T>
  template <class T>
  sum_of_weights_squared(const sum_of_weights_squared<T>& o) noexcept
      : sum_of_weights_squared(o.sum_of_weights_squared_) {}

  // Initialize to external sum of weights squared.
  sum_of_weights_squared(const_reference sum_w2) noexcept
      : sum_of_weights_squared_(sum_w2) {}

  // Increment by one.
  sum_of_weights_squared& operator++() {
    ++sum_of_weights_squared_;
    return *this;
  }

  // Increment by weight.
  sum_of_weights_squared& operator+=(const weight_type<value_type>& w) {
    sum_of_weights_squared_ += detail::square(w.value);
    return *this;
  }

  // Added another sum_of_weights_squared.
  sum_of_weights_squared& operator+=(const sum_of_weights_squared& rhs) {
    sum_of_weights_squared_ += rhs.sum_of_weights_squared_;
    return *this;
  }

  bool operator==(const sum_of_weights_squared& rhs) const noexcept {
    return sum_of_weights_squared_ == rhs.sum_of_weights_squared_;
  }

  bool operator!=(const sum_of_weights_squared& rhs) const noexcept {
    return !operator==(rhs);
  }

  // Return sum of weights squared.
  const_reference value() const noexcept { return sum_of_weights_squared_; }

  template <class Archive>
  void serialize(Archive& ar, unsigned /* version */) {
    ar& make_nvp("sum_of_weights_squared", sum_of_weights_squared_);
  }

private:
  ValueType sum_of_weights_squared_{};
};

} // namespace internal

/// Accumulates weighted boolean samples and computes the fraction of true samples.
template <class ValueType>
class weighted_fraction {
public:
  using value_type = ValueType;
  using const_reference = const value_type&;
  using fraction_type = fraction<ValueType>;
  using real_type = typename fraction_type::real_type;
  using interval_type = typename fraction_type::interval_type;

  weighted_fraction() noexcept = default;

  /// Initialize to external fraction and sum of weights squared.
  weighted_fraction(const fraction_type& f, const_reference sum_w2) noexcept
      : f_(f), sum_w2_(sum_w2) {}

  /// Convert the weighted_fraction class to a different type T.
  template <class T>
  operator weighted_fraction<T>() const noexcept {
    return weighted_fraction<T>(static_cast<fraction<T>>(f_),
                                static_cast<T>(sum_w2_.value()));
  }

  /// Insert boolean sample x with weight 1.
  void operator()(bool x) noexcept { operator()(weight(1), x); }

  /// Insert boolean sample x with weight w.
  void operator()(const weight_type<value_type>& w, bool x) noexcept {
    f_(w, x);
    sum_w2_ += w;
  }

  /// Add another weighted_fraction.
  weighted_fraction& operator+=(const weighted_fraction& rhs) noexcept {
    f_ += rhs.f_;
    sum_w2_ += rhs.sum_w2_;
    return *this;
  }

  bool operator==(const weighted_fraction& rhs) const noexcept {
    return f_ == rhs.f_ && sum_w2_ == rhs.sum_w2_;
  }

  bool operator!=(const weighted_fraction& rhs) const noexcept {
    return !operator==(rhs);
  }

  /// Return number of boolean samples that were true.
  const_reference successes() const noexcept { return f_.successes(); }

  /// Return number of boolean samples that were false.
  const_reference failures() const noexcept { return f_.failures(); }

  /// Return effective number of boolean samples.
  real_type count() const noexcept {
    return static_cast<real_type>(detail::square(f_.count())) / sum_w2_.value();
  }

  /// Return success weighted_fraction of boolean samples.
  real_type value() const noexcept { return f_.value(); }

  /// Return variance of the success weighted_fraction.
  real_type variance() const noexcept {
    return fraction_type::variance_for_p_and_n_eff(value(), count());
  }

  /// Return the sum of weights squared.
  value_type sum_of_weights_squared() const noexcept { return sum_w2_.value(); }

  /// Return standard interval with 68.3 % confidence level (Wilson score interval).
  interval_type confidence_interval() const noexcept {
    return confidence_interval(utility::wilson_interval<real_type>());
  }

  /// Return the Wilson score interval.
  interval_type confidence_interval(
      const utility::wilson_interval<real_type>& w) const noexcept {
    const real_type n_eff = count();
    const real_type p_hat = value();
    const real_type correction = w.third_order_correction(n_eff);
    return w.solve_for_neff_phat_correction(n_eff, p_hat, correction);
  }

  /// Return the fraction.
  const fraction_type& get_fraction() const noexcept { return f_; }

  /// Return the sum of weights squared.
  const value_type& sum_w2() const noexcept { return sum_w2_.value(); }

  template <class Archive>
  void serialize(Archive& ar, unsigned /* version */) {
    ar& make_nvp("fraction", f_);
    ar& make_nvp("sum_of_weights_squared", sum_w2_);
  }

private:
  fraction_type f_;
  internal::sum_of_weights_squared<ValueType> sum_w2_;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED

namespace std {
template <class T, class U>
/// Specialization for boost::histogram::accumulators::weighted_fraction.
struct common_type<boost::histogram::accumulators::weighted_fraction<T>,
                   boost::histogram::accumulators::weighted_fraction<U>> {
  using type = boost::histogram::accumulators::weighted_fraction<common_type_t<T, U>>;
};
} // namespace std

#endif

#endif
