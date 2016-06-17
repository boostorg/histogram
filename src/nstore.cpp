#include <boost/histogram/detail/nstore.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/int.hpp>
#include <stdexcept>
#include <new> // for std::bad_alloc

namespace boost {
namespace histogram {
namespace detail {

nstore::nstore() :
  size_(0),
  depth_(d1),
  buffer_(0)
{}

nstore::nstore(size_type s, depth_type d) :
  size_(s),
  depth_(d),
  buffer_(create(0))
{}

nstore&
nstore::operator+=(const nstore& o)
{
  if (size_ != o.size_)
    throw std::logic_error("sizes do not match");

  // make depth of lhs as large as rhs
  if (depth_ != o.depth_) {
    if (o.depth_ == dw)
      wconvert();
    else
      while (depth_ < o.depth_)
        grow();
  }

  // now add the content of lhs, grow as needed
  if (depth_ == dw) {
    for (size_type i = 0; i < size_; ++i)
      ((wtype*)buffer_)[i] += ((wtype*)o.buffer_)[i];
  }
  else {
    size_type i = size_;
    while (i--) {
      const uint64_t oi = o.ivalue(i);
      switch (static_cast<int>(depth_)) {
        #define BOOST_HISTOGRAM_NSTORE_ADD(D, T)            \
        case D: {                                           \
          T& b = ((T*)buffer_)[i];                          \
          if (T(std::numeric_limits<T>::max() - b) >= oi) { \
            b += oi;                                        \
            break;                                          \
          } else grow(); /* and fall through */             \
        }
        BOOST_HISTOGRAM_NSTORE_ADD(d1, uint8_t);
        BOOST_HISTOGRAM_NSTORE_ADD(d2, uint16_t);
        BOOST_HISTOGRAM_NSTORE_ADD(d4, uint32_t);
        BOOST_HISTOGRAM_NSTORE_ADD(d8, uint64_t);
        #undef BOOST_HISTOGRAM_NSTORE_ADD
        case dw: {
          ((wtype*)buffer_)[i] += wtype(oi);
          break;
        }
      }
    }
  }
  return *this;
}

bool
nstore::operator==(const nstore& o)
  const
{
  if (size_ != o.size_ || depth_ != o.depth_)
    return false;
  return std::memcmp(buffer_, o.buffer_, size_ * depth_) == 0;
}

double
nstore::value(size_type i)
  const
{
  if (depth_ == dw)
    return ((wtype*)buffer_)[i].w;
  return ivalue(i);
}

double
nstore::variance(size_type i)
  const
{
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_VARIANCE(D, T) \
    case D: return ((T*)buffer_)[i]
    BOOST_HISTOGRAM_NSTORE_VARIANCE(d1, uint8_t);
    BOOST_HISTOGRAM_NSTORE_VARIANCE(d2, uint16_t);
    BOOST_HISTOGRAM_NSTORE_VARIANCE(d4, uint32_t);
    BOOST_HISTOGRAM_NSTORE_VARIANCE(d8, uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_VARIANCE
    case dw: return ((wtype*)buffer_)[i].w2;
  }
  BOOST_ASSERT(!"never arrive here");
  return 0.0;
}

void*
nstore::create(void* buffer)
{
  void* b = buffer ? std::malloc(size_ * static_cast<int>(depth_)) :
                     std::calloc(size_,  static_cast<int>(depth_));
  if (!b)
    throw std::bad_alloc();
  if (buffer)
    std::memcpy(b, buffer, size_ * static_cast<int>(depth_));
  return b;
}

void
nstore::destroy()
{
  std::free(buffer_); buffer_ = 0;
}

void
nstore::grow()
{
  BOOST_ASSERT(size_ > 0);
  size_type i = size_;
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_GROW(D0, T0, D1, T1) \
    case D0:                                            \
      depth_ = D1;                                      \
      buffer_ = std::realloc(buffer_, size_ * static_cast<int>(depth_)); \
      if (!buffer_) throw std::bad_alloc();             \
      while (i--)                                       \
        ((T1*)buffer_)[i] = ((T0*)buffer_)[i];          \
      return
    BOOST_HISTOGRAM_NSTORE_GROW(d1, uint8_t, d2, uint16_t);
    BOOST_HISTOGRAM_NSTORE_GROW(d2, uint16_t, d4, uint32_t);
    BOOST_HISTOGRAM_NSTORE_GROW(d4, uint32_t, d8, uint64_t);
    BOOST_HISTOGRAM_NSTORE_GROW(d8, uint64_t, dw, wtype);
    #undef BOOST_HISTOGRAM_NSTORE_GROW
    case dw: BOOST_ASSERT(!"never arrive here");
  }
}

void
nstore::wconvert()
{
  BOOST_ASSERT(size_ > 0);
  BOOST_ASSERT(depth_ < dw);
  // realloc is safe if buffer_ is null
  buffer_ = std::realloc(buffer_, size_ * static_cast<int>(dw));
  if (!buffer_) throw std::bad_alloc();
  size_type i = size_;
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_CONVERT(D, T) \
    case D:                                      \
    while (i--)                                  \
      ((wtype*)buffer_)[i] = ((T*)buffer_)[i];   \
    break
    BOOST_HISTOGRAM_NSTORE_CONVERT(d1, uint8_t);
    BOOST_HISTOGRAM_NSTORE_CONVERT(d2, uint16_t);
    BOOST_HISTOGRAM_NSTORE_CONVERT(d4, uint32_t);
    BOOST_HISTOGRAM_NSTORE_CONVERT(d8, uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_CONVERT
    case dw: BOOST_ASSERT(!"never arrive here");
  }
  depth_ = dw;
}

uint64_t
nstore::ivalue(size_type i)
  const
{
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_IVALUE(D, T) \
    case D: return ((T*)buffer_)[i]
    BOOST_HISTOGRAM_NSTORE_IVALUE(d1, uint8_t);
    BOOST_HISTOGRAM_NSTORE_IVALUE(d2, uint16_t);
    BOOST_HISTOGRAM_NSTORE_IVALUE(d4, uint32_t);
    BOOST_HISTOGRAM_NSTORE_IVALUE(d8, uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_IVALUE
    case dw: BOOST_ASSERT(!"never arrive here");
  }
  return 0;
}

}
}
}
