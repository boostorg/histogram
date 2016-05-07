#include <boost/histogram/detail/nstore.hpp>
#include <stdexcept>
#include <new> // for std::bad_alloc

namespace boost {
namespace histogram {
namespace detail {

nstore::nstore() :
  size_(0),
  depth_(0),
  buffer_(0)
{}

nstore::nstore(size_type s, unsigned d) :
  size_(s),
  depth_(d),
  buffer_(create(0))
{
  BOOST_ASSERT(d == sizeof(uint8_t) ||
               d == sizeof(uint16_t) ||
               d == sizeof(uint32_t) ||
               d == sizeof(uint64_t) ||
               d == sizeof(wtype));
}

nstore&
nstore::operator+=(const nstore& o)
{
  if (size_ != o.size_)
    throw std::logic_error("sizes do not match");

  if (depth_ != o.depth_) {
    if (o.depth_ == sizeof(wtype))
      wconvert();
    else
      while (depth_ < o.depth_)
        grow();
  }

  if (depth_ == sizeof(wtype)) {
    for (size_type i = 0; i < size_; ++i)
      ((wtype*)buffer_)[i] += ((wtype*)o.buffer_)[i];
  }
  else {
    size_type i = 0;
    while (i < size_) {
      const uint64_t oi = o.ivalue(i);
      switch (depth_) {
        #define BOOST_HISTOGRAM_NSTORE_ADD(T)              \
        case sizeof(T): {                                  \
          T& b = ((T*)buffer_)[i];                         \
          if (T(std::numeric_limits<T>::max() - b) >= oi) { \
            b += oi;                                       \
            ++i;                                           \
            break;                                         \
          } else grow(); /* add fall through */            \
        }
        BOOST_HISTOGRAM_NSTORE_ADD(uint8_t);
        BOOST_HISTOGRAM_NSTORE_ADD(uint16_t);
        BOOST_HISTOGRAM_NSTORE_ADD(uint32_t);
        BOOST_HISTOGRAM_NSTORE_ADD(uint64_t);
        #undef BOOST_HISTOGRAM_NSTORE_ADD
        default: BOOST_ASSERT(!"invalid depth");
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
  if (depth_ == sizeof(wtype))
    return ((wtype*)buffer_)[i].w;
  return ivalue(i);
}

double
nstore::variance(size_type i)
  const
{
  switch (depth_) {
    case sizeof(wtype): return ((wtype*)buffer_)[i].w2;
    #define BOOST_HISTOGRAM_NSTORE_VARIANCE(T) \
    case sizeof(T): return ((T*)buffer_)[i]
    BOOST_HISTOGRAM_NSTORE_VARIANCE(uint8_t);
    BOOST_HISTOGRAM_NSTORE_VARIANCE(uint16_t);
    BOOST_HISTOGRAM_NSTORE_VARIANCE(uint32_t);
    BOOST_HISTOGRAM_NSTORE_VARIANCE(uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_VARIANCE
    default: BOOST_ASSERT(!"invalid depth");
  }
  return 0.0;
}

void*
nstore::create(void* buffer)
{
  if (size_ * depth_ == 0)
    return 0;
  void* b = buffer ? std::malloc(size_ * depth_) :
                     std::calloc(size_, depth_);
  if (!b)
    throw std::bad_alloc();
  if (buffer)
    std::memcpy(b, buffer, size_ * depth_);
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
  BOOST_ASSERT(depth_ > 0);
  BOOST_ASSERT(depth_ < sizeof(uint64_t));
  BOOST_ASSERT(buffer_ != 0);
  buffer_ = std::realloc(buffer_, size_ * 2 * depth_);
  if (!buffer_) throw std::bad_alloc();
  size_type i = size_;
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_GROW(T0, T1) \
    case sizeof(T0):                            \
    while (i--)                                 \
      ((T1*)buffer_)[i] = ((T0*)buffer_)[i];    \
    break
    BOOST_HISTOGRAM_NSTORE_GROW(uint8_t, uint16_t);
    BOOST_HISTOGRAM_NSTORE_GROW(uint16_t, uint32_t);
    BOOST_HISTOGRAM_NSTORE_GROW(uint32_t, uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_GROW
  }
  depth_ *= 2;
}

void
nstore::wconvert()
{
  BOOST_ASSERT(depth_ < sizeof(wtype));
  // realloc is safe if buffer_ is null
  buffer_ = std::realloc(buffer_, size_ * sizeof(wtype));
  if (!buffer_) throw std::bad_alloc();
  size_type i = size_;
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_CONVERT(T)  \
    case sizeof(T):                            \
    while (i--)                                \
      ((wtype*)buffer_)[i] = ((T*)buffer_)[i]; \
    break
    BOOST_HISTOGRAM_NSTORE_CONVERT(uint8_t);
    BOOST_HISTOGRAM_NSTORE_CONVERT(uint16_t);
    BOOST_HISTOGRAM_NSTORE_CONVERT(uint32_t);
    BOOST_HISTOGRAM_NSTORE_CONVERT(uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_CONVERT
    default: break; // no nuthin'
  }
  depth_ = sizeof(wtype);
}

uint64_t
nstore::ivalue(size_type i)
  const
{
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_IVALUE(T) \
    case sizeof(T): return ((T*)buffer_)[i]
    BOOST_HISTOGRAM_NSTORE_IVALUE(uint8_t);
    BOOST_HISTOGRAM_NSTORE_IVALUE(uint16_t);
    BOOST_HISTOGRAM_NSTORE_IVALUE(uint32_t);
    BOOST_HISTOGRAM_NSTORE_IVALUE(uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_IVALUE
    default: BOOST_ASSERT(!"invalid depth");
  }
  return 0;
}

}
}
}
