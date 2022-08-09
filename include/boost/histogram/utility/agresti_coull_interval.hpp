// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_AGRESTI_COULL_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_AGRESTI_COULL_INTERVAL_HPP

#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <cmath>
#include <utility>

namespace boost {
namespace histogram {
namespace utility {

template <class ValueType>
class agresti_coull_interval : public binomial_proportion_interval<ValueType> {
  using base_t = binomial_proportion_interval<ValueType>;

public:
  using value_type = typename base_t::value_type;
  using interval_type = typename base_t::interval_type;

  explicit agresti_coull_interval(deviation d = deviation{1.0}) noexcept
      : z_{static_cast<double>(d)} {}

  interval_type operator()(value_type successes, value_type failures) const noexcept {
    const value_type half{0.5}, one{1.0}, zsq = z_ * z_;
    const value_type ns = successes, nf = failures;
    const value_type n = ns + nf;
    const value_type n_ac = n + zsq;
    const value_type a = (ns + zsq * half) / n_ac;
    const value_type b = z_ * std::sqrt((one - a) * a / n_ac);
    return std::make_pair(a - b, a + b);
  }

private:
  double z_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif