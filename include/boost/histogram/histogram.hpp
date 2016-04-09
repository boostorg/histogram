#ifndef _BOOST_HISTOGRAM_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/histogram/histogram_base.hpp>
#include <boost/histogram/detail/nstore.hpp>
#include <boost/preprocessor.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/assert.hpp>

namespace boost {
namespace histogram {

class histogram : public histogram_base {
public:
  histogram() {}

  histogram(const histogram& o) :
    histogram_base(o),
    data_(o.data_)
  {}

  explicit
  histogram(const axes_type& axes) :
    histogram_base(axes),
    data_(field_count())
  {}

#define BOOST_NHISTOGRAM_CTOR(z, n, unused)                        \
  histogram( BOOST_PP_ENUM_PARAMS_Z(z, n, const axis_type& a) ) : \
    histogram_base( BOOST_PP_ENUM_PARAMS_Z(z, n, a) ),             \
    data_(field_count())                                           \
  {}

// generates constructors taking 1 to AXIS_LIMIT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_NHISTOGRAM_CTOR, nil)

  unsigned depth() const { return data_.depth(); }

  double sum() const;

  template <typename Container>
  inline
  void fill(const Container& v)
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

  template <typename Container>
  inline
  void wfill(const Container& v, double w)
  {
    BOOST_ASSERT(v.size() == dim());
    const size_type k = pos(v);
    if (k != uintmax_t(-1))
      data_.increase(k, w);
  }

  // C-style call
  inline
  void wfill(unsigned n, const double* v, double w)
  {
    BOOST_ASSERT(n == dim());
    const size_type k = pos(v);
    if (k != uintmax_t(-1))
      data_.increase(k, w);
  }

#define BOOST_NHISTOGRAM_WFILL(z, n, unused)                     \
  inline                                                         \
  void wfill( BOOST_PP_ENUM_PARAMS_Z(z, n, double x), double w ) \
  {                                                              \
    const double buffer[n] = { BOOST_PP_ENUM_PARAMS(n, x) };     \
    wfill(n, buffer, w); /* size is checked here */              \
  }

// generates wfill functions taking 1 to AXIS_LIMT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_NHISTOGRAM_WFILL, nil)  

  template <typename Container>
  double value(const Container& idx)
    const
  {
    BOOST_ASSERT(idx.size() == dim());
    return data_.value(linearize(idx));
  }

  // C-style call
  double value(unsigned n, const int* idx)
    const
  {
    BOOST_ASSERT(n == dim());
    return data_.value(linearize(idx));
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

  template <typename Container>
  double variance(const Container& idx)
    const
  {
    BOOST_ASSERT(idx.size() == dim());
    return data_.variance(linearize(idx));
  }

  // C-style call
  double variance(unsigned n, const int* idx)
    const
  {
    BOOST_ASSERT(n == dim());
    return data_.variance(linearize(idx));
  }

#define BOOST_NHISTOGRAM_VARIANCE(z, n, unused)             \
  double variance( BOOST_PP_ENUM_PARAMS_Z(z, n, int i) )    \
    const                                                   \
  {                                                         \
    const int idx[n] = { BOOST_PP_ENUM_PARAMS_Z(z, n, i) }; \
    return variance(n, idx); /* size is checked here */     \
  }

// generates variance functions taking 1 to AXIS_LIMT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_NHISTOGRAM_VARIANCE, nil)  

  bool operator==(const histogram& o) const 
  { return histogram_base::operator==(o) &&
           data_ == o.data_; }

  histogram& operator+=(const histogram& o)
  { data_ += o.data_; return *this; }

private:
  detail::nstore data_;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    ar & serialization::base_object<histogram_base>(*this);
    ar & data_;
  }

  friend class buffer_access;
};

histogram operator+(const histogram& a, const histogram& b) {
  histogram result(a);
  return result += b;
}

}
}

#endif
