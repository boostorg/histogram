#ifndef _BOOST_HISTOGRAM_NHISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_NHISTOGRAM_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/histogram/histogram_base.hpp>
#include <boost/histogram/nstore.hpp>
#include <boost/preprocessor.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/assert.hpp>
#include <boost/concept/requires.hpp>
#include <ostream>
#include <vector>

namespace boost {
namespace histogram {

class nhistogram : public histogram_base {
public:
  nhistogram() {}

  nhistogram(const nhistogram& o) :
    histogram_base(o),
    data_(o.data_)
  {}

  explicit
  nhistogram(const axes_type& axes) :
    histogram_base(axes),
    data_(field_count())
  {}

#define BOOST_NHISTOGRAM_CTOR(z, n, unused)                        \
  nhistogram( BOOST_PP_ENUM_PARAMS_Z(z, n, const axis_type& a) ) : \
    histogram_base( BOOST_PP_ENUM_PARAMS_Z(z, n, a) ),             \
    data_(field_count())                                           \
  {}

// generates constructors taking 1 to AXIS_LIMIT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_NHISTOGRAM_CTOR, nil)

  double sum() const;

  template <typename T>
  inline
  void fill(const T& v)
  {
    BOOST_ASSERT(v.size() == dim());
    const size_type k = pos(v);
    if (k != uintmax_t(-1))
      data_.increase(k);
  }

  // C-style call
  inline
  void fill(unsigned n, const double* v)
  {
    BOOST_ASSERT(n == dim());
    const size_type k = pos(v);
    if (k != uintmax_t(-1))
      data_.increase(k);
  }

#define BOOST_NHISTOGRAM_FILL(z, n, unused)                  \
  inline                                                     \
  void fill( BOOST_PP_ENUM_PARAMS_Z(z, n, double x) )        \
  {                                                          \
    const double buffer[n] = { BOOST_PP_ENUM_PARAMS(n, x) }; \
    fill(n, buffer); /* size is checked here */              \
  }

// generates fill functions taking 1 to AXIS_LIMT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_NHISTOGRAM_FILL, nil)  

  template <typename Array>
  double value(const Array& idx)
    const
  {
    BOOST_ASSERT(idx.size() == dim());
    return data_.read(linearize(idx));
  }

  // C-style call
  double value(unsigned n, const int* idx)
    const
  {
    BOOST_ASSERT(n == dim());
    return data_.read(linearize(idx));
  }

#define BOOST_NHISTOGRAM_VALUE(z, n, unused)                \
  double value( BOOST_PP_ENUM_PARAMS_Z(z, n, int i) )       \
    const                                                   \
  {                                                         \
    const int idx[n] = { BOOST_PP_ENUM_PARAMS_Z(z, n, i) }; \
    return value(n, idx); /* size is checked here */        \
  }

// generates value functions taking 1 to AXIS_LIMT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_NHISTOGRAM_VALUE, nil)  

  bool operator==(const nhistogram& o) const 
  { return histogram_base::operator==(o) &&
           data_ == o.data_; }

  nhistogram& operator+=(const nhistogram& o)
  { data_ += o.data_; return *this; }

  // needed by boost::python interface to pass internal data to numpy
  const nstore& data() const { return data_; }

private:
  nstore data_;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    using namespace serialization;
    ar & boost::serialization::base_object<histogram_base>(*this);
    ar & data_;
  }
};

nhistogram operator+(const nhistogram& a, const nhistogram& b) {
  nhistogram result(a);
  return result += b;
}

}
}

#endif
