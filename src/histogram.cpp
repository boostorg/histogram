#include <boost/histogram/histogram.hpp>
#include <ostream>

namespace boost {
namespace histogram {

histogram::histogram(const histogram& o) :
    basic_histogram(o),
    data_(o.data_)
{}

histogram& histogram::operator=(BOOST_COPY_ASSIGN_REF(histogram) o)
{
    if (this != &o) {
        basic_histogram::operator=(static_cast<const basic_histogram&>(o));
        data_ = o.data_;
    }
    return *this;
}

histogram::histogram(BOOST_RV_REF(histogram) o) :
    basic_histogram(::boost::move(static_cast<basic_histogram&>(o))),
    data_(::boost::move(o.data_))
{}

histogram& histogram::operator=(BOOST_RV_REF(histogram) o)
{
    if (this != &o) {
        basic_histogram::operator=(::boost::move(static_cast<basic_histogram&>(o)));
        data_ = ::boost::move(o.data_);
    }
    return *this;
}

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
