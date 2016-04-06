#include <boost/histogram/nhistogram.hpp>
#include <ostream>

namespace boost {
namespace histogram {

double
nhistogram::sum()
    const
{
    double result = 0.0;
    for (size_type i = 0, n = field_count(); i < n; ++i)
      result += data_.read(i);
    return result;
}

}
}
