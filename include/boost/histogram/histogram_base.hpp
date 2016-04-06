#ifndef _BOOST_HISTOGRAM_BASE_HPP_
#define _BOOST_HISTOGRAM_BASE_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/histogram/visitors.hpp>
#include <boost/cstdint.hpp>
#include <boost/preprocessor.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/type_traits/is_same.hpp>

#include <bitset>

#define BOOST_HISTOGRAM_AXIS_LIMIT 16

namespace boost {
namespace histogram {

// holds an array of axis and computes the internal index
class histogram_base {
public:
  typedef uint64_t size_type;

  histogram_base(const histogram_base&);
  histogram_base& operator=(const histogram_base&);
  ~histogram_base() {}

  unsigned dim() const { return axes_.size(); }

  template <typename T>
  T& axis(unsigned i) 
  { if (is_same<T, axis_type>::value) return axes_[i];
    return boost::get<T&>(axes_[i]); }

  template <typename T>
  const T& axis(unsigned i) const
  { if (is_same<T, axis_type>::value) return axes_[i];
    return boost::get<const T&>(axes_[i]); }

  unsigned bins(unsigned i) const { return size_[i]; }
  unsigned shape(unsigned i) const { return size_[i] + 2 * uoflow_[i]; }

protected:
  histogram_base() {}

  explicit histogram_base(const axes_type& axes);

  explicit histogram_base(const axis_type& a) : axes_(1, a) { update_buffers(); }

#define BOOST_HISTOGRAM_BASE_APPEND(z, n, unused) axes_.push_back(a ## n);
#define BOOST_HISTOGRAM_BASE_CTOR(z, n, unused)                      \
  histogram_base( BOOST_PP_ENUM_PARAMS_Z(z, n, const axis_type& a) ) \
  {                                                                  \
    axes_.reserve(n);                                                \
    BOOST_PP_REPEAT(n, BOOST_HISTOGRAM_BASE_APPEND, unused)          \
    update_buffers();                                                \
  }

// generates constructors taking 2 to AXIS_LIMIT arguments
BOOST_PP_REPEAT_FROM_TO(2, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_HISTOGRAM_BASE_CTOR, nil)

  bool operator==(const histogram_base&) const;
  bool operator!=(const histogram_base& o) const
  { return !operator==(o); }

  template <typename Array>
  inline
  size_type pos(const Array& v) const {
    int32_t idx[BOOST_HISTOGRAM_AXIS_LIMIT];
    for (unsigned i = 0, n = axes_.size(); i < n; ++i)
      idx[i] = apply_visitor(detail::index_visitor(v[i]), axes_[i]);
    return linearize(idx);
  }

  inline
  size_type linearize(const int* idx) const {
    size_type stride = 1, k = 0, i = axes_.size();
    while (i) {
      --i;
      const int size = size_[i];
      const int range = size + 2 * uoflow_[i];
      int j = idx[i];
      // the following three lines work for any overflow setting
      j += (j < 0) * (size + 2); // wrap around if j < 0
      if (j >= range)
        return size_type(-1); // indicate out of range
      k += j * stride;
      stride *= range;
    }
    return k;
  }

  // compute the number of fields needed for storage
  size_type field_count() const;

private:
  axes_type axes_;

  // internal buffers
  int32_t size_[BOOST_HISTOGRAM_AXIS_LIMIT];
  std::bitset<BOOST_HISTOGRAM_AXIS_LIMIT> uoflow_;

  void update_buffers(); ///< fills size_ and uoflow_

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    using namespace serialization;
    ar & axes_;
    if (Archive::is_loading::value)
      update_buffers();
  }
};

}
}

#endif
