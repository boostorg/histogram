// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_EFFICIENCY_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_EFFICIENCY_HPP

#include <boost/core/nvp.hpp>
#include <boost/histogram/detail/atomic_number.hpp>
#include <boost/histogram/fwd.hpp> // for efficiency<>
#include <type_traits>
#include <boost/format.hpp>
#include <boost/histogram.hpp>
#include <cmath>

enum interval_type {
    Wald=0,
    Wilson=1,
    Jeffreys=2,
    ClopperPearson=3,
    AgrestiCoull=4
};

struct efficiency {
    // return value is ignored, so we use void
    void operator()(bool x) {
        if (x) ++n_success_;
        else ++n_failure_;
    }

    double value() const { return n_success_/(n_success_+n_failure_); }
    
    double variance() const { return n_success_*n_failure_*(n_success_+n_failure_); }

    double confidence_interval(interval_type CI_type = Wald, int conf_level = 95) const {
        double z;
        if (conf_level >= 99) {z = 2.576; }
        else if (conf_level >= 98 && conf_level < 99) {z = 2.326; }
        else if (conf_level >= 95 && conf_level < 98) {z = 1.96; }
        else {z = 1.645; }
        
        switch(CI_type)
        {
            case Wald: return (z*pow((n_failure_*n_success_), 0.5)) / pow(n_failure_+n_success_, 1.5);
            case Wilson: return (z/(n_failure_+n_success_+pow(z, 2)))*pow(((n_failure_*n_success_)/(n_failure_ + n_success_))+(pow(z,2)/4), 0.5);
            case Jeffreys: return 0; // implement if needed
            case ClopperPearson: return 0; // implement if needed
            case AgrestiCoull: return 0; // implement if needed
        };
        return 0;
    }

    double n_success_ = 0;
    double n_failure_ = 0;
};

#endif
