// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_MEAN_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_MEAN_HPP

#include <boost/histogram/weight.hpp>

namespace boost {
namespace histogram {
namespace accumulators {

/// Holds sum of weights, sum of weighted values, and sum of weighted values squared
template <typename RealType = double>
class mean {
public:
  mean() = default;
  mean(const mean&) = default;
  mean(mean&&) = default;
  mean& operator=(const mean&) = default;
  mean& operator=(mean&&) = default;

  mean(const RealType& w, const RealType& wx, const RealType& wxx) noexcept
      : w_(w), wx_(wx), wxx_(wxx) {}

  void operator()(const RealType& x) noexcept {
    ++w_;
    wx_ += x;
    wxx_ += x * x;
  }

  template <typename T>
  void operator()(const ::boost::histogram::weight_type<T>& w,
                  const RealType& x) noexcept {
    w_ += w.value;
    wx_ += w.value * x;
    wxx_ += w.value * x * x;
  }

  template <typename T>
  mean& operator+=(const mean<T>& rhs) {
    w_ += static_cast<RealType>(rhs.w_);
    wx_ += static_cast<RealType>(rhs.wx_);
    wxx_ += static_cast<RealType>(rhs.wxx_);
    return *this;
  }

  mean& operator*=(const RealType& s) noexcept {
    wx_ *= s;
    wxx_ *= s * s;
    return *this;
  }

  bool operator==(const mean& rhs) const noexcept {
    return w_ == rhs.w_ && wx_ == rhs.wx_ && wxx_ == rhs.wxx_;
  }

  bool operator!=(const mean& rhs) const noexcept { return !operator==(rhs); }

  template <typename T>
  bool operator==(const mean<T>& rhs) const noexcept {
    return w_ == rhs.w_ && wx_ == rhs.wx_ && wxx_ == rhs.wxx_;
  }

  template <typename T>
  bool operator!=(const mean<T>& rhs) const noexcept {
    return !operator==(rhs);
  }

  const RealType& sum() const noexcept { return w_; }
  RealType value() const noexcept { return wx_ / w_; }
  RealType variance() const noexcept {
    return wxx_ / (w_ - 1) - w_ / (w_ - 1) * value() * value();
  }

  template <class Archive>
  void serialize(Archive&, unsigned /* version */);

private:
  RealType w_ = 0, wx_ = 0, wxx_ = 0;
};

template <typename T>
mean<T> operator+(const mean<T>& a, const mean<T>& b) noexcept {
  mean<T> c = a;
  return c += b;
}

template <typename T>
mean<T>&& operator+(mean<T>&& a, const mean<T>& b) noexcept {
  a += b;
  return std::move(a);
}

template <typename T>
mean<T>&& operator+(const mean<T>& a, mean<T>&& b) noexcept {
  return operator+(std::move(b), a);
}

template <typename T>
mean<T> operator+(const mean<T>& a, const T& b) noexcept {
  auto r = a;
  return r += b;
}

template <typename T>
mean<T> operator+(const T& a, const mean<T>& b) noexcept {
  auto r = b;
  return r += a;
}

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif
