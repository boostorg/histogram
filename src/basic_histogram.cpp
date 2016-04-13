#include <boost/histogram/basic_histogram.hpp>
#include <boost/histogram/axis.hpp>
#include <stdexcept>

namespace boost {
namespace histogram {

basic_histogram::basic_histogram(const basic_histogram& o) :
  axes_(o.axes_)
{
  update_buffers();
}

basic_histogram::basic_histogram(const axes_type& axes) :
  axes_(axes)
{
  if (axes_.size() > BOOST_HISTOGRAM_AXIS_LIMIT)
    throw std::invalid_argument("too many axes");
  update_buffers();
}

basic_histogram&
basic_histogram::operator=(const basic_histogram& o)
{
  axes_ = o.axes_;
  update_buffers();
  return *this;
}

bool
basic_histogram::operator==(const basic_histogram& o)
  const
{
  if (axes_.size() != o.axes_.size())
    return false;
  for (unsigned i = 0; i < axes_.size(); ++i) {
    if (!apply_visitor(detail::cmp_visitor(), axes_[i], o.axes_[i]))
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
    fc *= apply_visitor(detail::fields_visitor(), axes_[i]);
  return fc;
}

void
basic_histogram::update_buffers()
{
  for (unsigned i = 0, n = axes_.size(); i < n; ++i) {
    size_[i] = apply_visitor(detail::bins_visitor(), axes_[i]);
    uoflow_[i] = apply_visitor(detail::uoflow_visitor(), axes_[i]);
  }
}

}
}
