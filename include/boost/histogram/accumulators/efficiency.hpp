// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_EFFICIENCY_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_EFFICIENCY_HPP

#include <boost/core/nvp.hpp>
#include <boost/histogram/fwd.hpp> // for efficiency<>
#include <boost/histogram/detail/probit.hpp>
#include <boost/histogram/detail/waldInterval.hpp>
#include <boost/histogram/detail/wilsonInterval.hpp>
#include <utility>
#include <limits>

enum class interval_type {
    wald,
    wilson,
    jeffreys,
    agrestiCoull,
    clopperPearson
};

struct efficiency {
    void operator()(bool x) {
        if (x) ++n_success_;
        else ++n_failure_;
    }

    double value() const { return n_success_/(n_success_+n_failure_); }
    
    double variance() const { return n_success_*n_failure_*(n_success_+n_failure_); } // Source: Variance from Binomial Distribution | Wikipedia

    std::pair<double, double> confidence_interval(interval_type type = interval_type::wald, double cl = 0.683) const {
        double z = detail::probit(cl);
        double p = n_success_/(n_success_+n_failure_);
        switch(type)
        {
            case interval_type::wald: return std::make_pair((p-(detail::waldInterval(n_failure_, n_success_, z))), (p+(detail::waldInterval(n_failure_, n_success_, z))));
            case interval_type::wilson: return std::make_pair((p-(detail::wilsonInterval(n_failure_, n_success_, z))), (p+(detail::wilsonInterval(n_failure_, n_success_, z))));
            case interval_type::jeffreys: return std::make_pair(0, 0); // implement if needed
            case interval_type::clopperPearson: return std::make_pair(0, 0); // implement if needed
            case interval_type::agrestiCoull: return std::make_pair(0, 0); // implement if needed
        };
        return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()); // code should never arrive here
    }

    double n_success_ = 0;
    double n_failure_ = 0;
};

#endif
