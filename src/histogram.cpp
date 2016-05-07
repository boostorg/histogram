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
