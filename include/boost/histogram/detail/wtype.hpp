#ifndef _BOOST_HISTOGRAM_DETAIL_WTYPE_HPP_
#define _BOOST_HISTOGRAM_DETAIL_WTYPE_HPP_

#include <boost/cstdint.hpp>
#include <ostream>

namespace boost {
namespace histogram {
namespace detail {

struct wtype {
  double w, w2;
  wtype() : w(0), w2(0) {}  
  wtype(const wtype& o) : w(o.w), w2(o.w2) {}
  wtype(uint64_t i) : w(i), w2(i) {}
  wtype& operator+=(const wtype& o)
  { w += o.w; w2 += o.w2; return *this; }
  wtype& operator+=(double v)
  { w += v; w2 += v*v; return *this; }
  wtype& operator=(uint64_t i)
  { w = i; w2 = i; return *this; }
  bool operator==(uint64_t i) const
  { return w == i; }
  bool operator!=(uint64_t i) const
  { return w != i; }
  bool operator==(const wtype& o) const
  { return w == o.w && w2 == o.w2; }
  bool operator!=(const wtype& o) const
  { return w != o.w || w2 != o.w2; }
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  { ar & w; ar & w2; }
};

static
inline
std::ostream& operator<<(std::ostream& os, const wtype& w)
{
  os << '(' << w.w << ',' << w.w2 << ')';
  return os;
}

}
}
}

#endif
