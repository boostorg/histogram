#include <boost/histogram/basic_histogram.hpp>
#include <boost/histogram/axis.hpp>
#include <stdexcept>

namespace boost {
namespace histogram {

basic_histogram::basic_histogram(const basic_histogram& o) :
  axes_(o.axes_)
{}

basic_histogram&
basic_histogram::operator=(BOOST_COPY_ASSIGN_REF(basic_histogram) o)
{
  if (this != &o) {
    axes_ = o.axes_;
  }
  return *this;
}

basic_histogram::basic_histogram(BOOST_RV_REF(basic_histogram) o) :
  axes_(::boost::move(o.axes_))
{}

basic_histogram&
basic_histogram::operator=(BOOST_RV_REF(basic_histogram) o)
{
  if (this != &o) {
    axes_ = ::boost::move(o.axes_);
  }
  return *this;
}

basic_histogram::basic_histogram(const axes_type& axes) :
  axes_(axes)
{
  if (axes_.size() > BOOST_HISTOGRAM_AXIS_LIMIT)
    throw std::invalid_argument("too many axes");
}

bool
basic_histogram::operator==(const basic_histogram& o)
  const
{
  if (axes_.size() != o.axes_.size())
    return false;
  for (unsigned i = 0; i < axes_.size(); ++i) {
    if (!apply_visitor(visitor::cmp(), axes_[i], o.axes_[i]))
      return false;
  }
  return true;
}

basic_histogram::size_type
basic_histogram::field_count()
  const
{
  size_type fc = 1;
  for (unsigned i = 0, n = axes_.size(); i < n; ++i)
    fc *= shape(i);
  return fc;
}

}
}
