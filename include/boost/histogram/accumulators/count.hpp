// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_COUNT_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_COUNT_HPP

#include <atomic>
#include <boost/core/nvp.hpp>
#include <boost/histogram/fwd.hpp> // for count<>
#include <type_traits>             // for std::common_type

namespace boost {
namespace histogram {
namespace detail {

template <class Derived, bool B>
struct atomic_float_ext {};

template <class Derived>
struct atomic_float_ext<Derived, true> {
  Derived& operator++() noexcept {
    this->operator+=(static_cast<typename Derived::value_type>(1));
    return static_cast<Derived&>(*this);
  }

  Derived& operator+=(typename Derived::const_reference x) noexcept {
    auto expected = this->load();
    // if another tread changed expected value, compare_exchange returns false
    // and updates expected; we then loop and try to update again;
    // see https://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange
    while (!this->compare_exchange_weak(expected, expected + x))
      ;
    return static_cast<Derived&>(*this);
  }
};

// copyable arithmetic type with atomic operator++ and operator+=,
// works on floating point numbers already in C++14
template <class T>
struct atomic : std::atomic<T>,
                atomic_float_ext<atomic<T>, std::is_floating_point<T>::value> {
  static_assert(std::is_arithmetic<T>(), "");

  using value_type = T;
  using const_reference = const T&;

  using std::atomic<T>::atomic;

  atomic() noexcept = default;
  atomic(const atomic& o) noexcept : std::atomic<T>{o.load()} {}
  atomic& operator=(const atomic& o) noexcept {
    this->store(o.load());
    return *this;
  }
};
} // namespace detail

namespace accumulators {

/**
  Wraps a C++ builtin arithmetic type which is optionally thread-safe.

  This adaptor optionally uses atomic operations to make concurrent increments and
  additions safe for the stored arithmetic value, which can be an integer or floating
  point. Warning: Assignment is not thread-safe,so don't assign concurrently.

  Furthermore, this wrapper class can be used as a base class by users to add
  arbitrary metadata to each bin of a histogram.

  When weighted samples are accumulated and high precision is required, use
  `accumulators::sum` instead (at the cost of lower performance). If a local variance
  estimate for the weight distribution should be computed as well (generally needed for a
  detailed statistical analysis), use `accumulators::weighted_sum`.
*/
template <class ValueType, bool ThreadSafe>
class count {
  using internal_type =
      std::conditional_t<ThreadSafe, detail::atomic<ValueType>, ValueType>;

public:
  using value_type = ValueType;
  using const_reference = const value_type&;

  count() noexcept = default;

  /// Initialize count to value and allow implicit conversion
  count(const_reference value) noexcept : value_{value} {}

  /// Allow implicit conversion from other count
  template <class T, bool B>
  count(const count<T, B>& c) noexcept : count{c.value()} {}

  /// Increment count by one
  count& operator++() noexcept {
    ++value_;
    return *this;
  }

  /// Increment count by value
  count& operator+=(const_reference value) noexcept {
    value_ += value;
    return *this;
  }

  /// Add another count
  count& operator+=(const count& s) noexcept {
    value_ += s.value_;
    return *this;
  }

  /// Scale by value
  count& operator*=(const_reference value) noexcept {
    value_ *= value;
    return *this;
  }

  bool operator==(const count& rhs) const noexcept { return value_ == rhs.value_; }

  bool operator!=(const count& rhs) const noexcept { return !operator==(rhs); }

  /// Return count
  const_reference value() const noexcept { return value_; }

  // conversion to value_type must be explicit
  explicit operator value_type() const noexcept { return value_; }

  template <class Archive>
  void serialize(Archive& ar, unsigned /* version */) {
    ar& make_nvp("value", value_);
  }

  // begin: extra operators to make count behave like a regular number

  count& operator*=(const count& rhs) noexcept {
    value_ *= rhs.value_;
    return *this;
  }

  count operator*(const count& rhs) const noexcept {
    count x = *this;
    x *= rhs;
    return x;
  }

  count& operator/=(const count& rhs) noexcept {
    value_ /= rhs.value_;
    return *this;
  }

  count operator/(const count& rhs) const noexcept {
    count x = *this;
    x /= rhs;
    return x;
  }

  bool operator<(const count& rhs) const noexcept { return value_ < rhs.value_; }

  bool operator>(const count& rhs) const noexcept { return value_ > rhs.value_; }

  bool operator<=(const count& rhs) const noexcept { return value_ <= rhs.value_; }

  bool operator>=(const count& rhs) const noexcept { return value_ >= rhs.value_; }

  // end: extra operators

private:
  internal_type value_{};
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED
namespace std {
template <class T, class U, bool B1, bool B2>
struct common_type<boost::histogram::accumulators::count<T, B1>,
                   boost::histogram::accumulators::count<U, B2>> {
  using type = boost::histogram::accumulators::count<common_type_t<T, U>, (B1 || B2)>;
};
} // namespace std
#endif

#endif
