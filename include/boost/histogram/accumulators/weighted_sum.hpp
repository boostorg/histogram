// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_SUM_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_SUM_HPP

#include <boost/core/nvp.hpp>
#include <boost/histogram/fwd.hpp> // for weighted_sum<>
#include <type_traits>

namespace boost {
namespace histogram {
namespace accumulators {

/// Holds sum of weights and its variance estimate
template <class RealType>
class weighted_sum {
public:
  using value_type = RealType;
  using const_reference = const RealType&;

  weighted_sum() = default;
  explicit weighted_sum(const RealType& value) noexcept
      : sum_of_weights_(value), sum_of_weights_squared_(value) {}
  weighted_sum(const RealType& value, const RealType& variance) noexcept
      : sum_of_weights_(value), sum_of_weights_squared_(variance) {}

  /// Increment by one.
  weighted_sum& operator++() { return operator+=(1); }

  /// Increment by value.
  template <class T>
  weighted_sum& operator+=(const T& value) {
    sum_of_weights_ += value;
    sum_of_weights_squared_ += value * value;
    return *this;
  }

  /// Added another weighted sum.
  template <class T>
  weighted_sum& operator+=(const weighted_sum<T>& rhs) {
    sum_of_weights_ += static_cast<RealType>(rhs.sum_of_weights_);
    sum_of_weights_squared_ += static_cast<RealType>(rhs.sum_of_weights_squared_);
    return *this;
  }

  /// Scale by value.
  weighted_sum& operator*=(const RealType& x) {
    sum_of_weights_ *= x;
    sum_of_weights_squared_ *= x * x;
    return *this;
  }

  bool operator==(const RealType& rhs) const noexcept {
    return sum_of_weights_ == rhs && sum_of_weights_squared_ == rhs;
  }

  template <class T>
  bool operator==(const weighted_sum<T>& rhs) const noexcept {
    return sum_of_weights_ == rhs.sum_of_weights_ &&
           sum_of_weights_squared_ == rhs.sum_of_weights_squared_;
  }

  template <class T>
  bool operator!=(const T& rhs) const noexcept {
    return !operator==(rhs);
  }

  /// Return value of the sum.
  const_reference value() const noexcept { return sum_of_weights_; }

  /// Return estimated variance of the sum.
  const_reference variance() const noexcept { return sum_of_weights_squared_; }

  // lossy conversion must be explicit
  explicit operator const_reference() const { return sum_of_weights_; }

  template <class Archive>
  void serialize(Archive& ar, unsigned /* version */) {
    ar& make_nvp("sum_of_weights", sum_of_weights_);
    ar& make_nvp("sum_of_weights_squared", sum_of_weights_squared_);
  }

private:
  value_type sum_of_weights_ = value_type();
  value_type sum_of_weights_squared_ = value_type();
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED
namespace std {
template <class T, class U>
struct common_type<boost::histogram::accumulators::weighted_sum<T>,
                   boost::histogram::accumulators::weighted_sum<U>> {
  using type = boost::histogram::accumulators::weighted_sum<common_type_t<T, U>>;
};

template <class T, class U>
struct common_type<boost::histogram::accumulators::weighted_sum<T>, U> {
  using type = boost::histogram::accumulators::weighted_sum<common_type_t<T, U>>;
};

template <class T, class U>
struct common_type<T, boost::histogram::accumulators::weighted_sum<U>> {
  using type = boost::histogram::accumulators::weighted_sum<common_type_t<T, U>>;
};
} // namespace std
#endif

#endif
