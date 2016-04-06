#ifndef _BOOST_HISTOGRAM_NHISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_NHISTOGRAM_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/histogram/histogram_base.hpp>
#include <boost/histogram/nstore.hpp>
#include <boost/preprocessor.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/shared_ptr.hpp>
#include <ostream>
#include <vector>

namespace boost {
namespace histogram {

class nhistogram : public histogram_base {
public:
  nhistogram() {}

  explicit
  nhistogram(const axes_type& axes) :
    histogram_base(axes),
    data_(field_count())
  {}

  explicit
  nhistogram(const axis_type& a) :
    histogram_base(a),
    data_(field_count())
  {}

#define BOOST_NHISTOGRAM_CTOR(z, n, unused)                        \
  nhistogram( BOOST_PP_ENUM_PARAMS_Z(z, n, const axis_type& a) ) : \
    histogram_base( BOOST_PP_ENUM_PARAMS_Z(z, n, a) ),             \
    data_(field_count())                                           \
  {}

// generates constructors taking 2 to AXIS_LIMIT arguments
BOOST_PP_REPEAT_FROM_TO(2, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_NHISTOGRAM_CTOR, nil)

  double sum() const;

  template <typename Array>
  inline
  void fill(const Array& v)
  {
    const size_type k = pos(v);
    if (k != uintmax_t(-1))
      data_.increase(k);
  }

  void fill(double x,
    BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(BOOST_PP_DEC(BOOST_HISTOGRAM_AXIS_LIMIT),
                                        double x, 0.0))
  {
    const double buffer[BOOST_HISTOGRAM_AXIS_LIMIT] = {
      x, BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(BOOST_HISTOGRAM_AXIS_LIMIT), x)
    };
    fill(buffer);
  }

  template <typename Array>
  inline
  size_type operator()(const Array& idx)
    const
  { return data_.read(linearize(idx)); }

  size_type operator()(int i,
    BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(BOOST_PP_DEC(BOOST_HISTOGRAM_AXIS_LIMIT),
                                        int i, 0))
    const
  {
    const int32_t idx[BOOST_HISTOGRAM_AXIS_LIMIT] = {
      i, BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(BOOST_HISTOGRAM_AXIS_LIMIT), i)
    };
    return operator()(idx);
  }

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
