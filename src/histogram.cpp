// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/histogram.hpp>
#include <ostream>

namespace boost {
namespace histogram {

histogram::histogram(const axes_type& axes) :
    basic_histogram(axes),
    data_(field_count())
{}

double
histogram::sum()
    const
{
    double result = 0.0;
    for (size_type i = 0, n = field_count(); i < n; ++i)
      result += data_.value(i);
    return result;
}

}
}
