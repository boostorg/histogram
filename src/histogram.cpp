#include <boost/histogram/histogram.hpp>
#include <ostream>

namespace boost {
namespace histogram {

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
