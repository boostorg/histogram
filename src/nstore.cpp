#include <boost/histogram/detail/nstore.hpp>
#include <boost/cstdint.hpp>
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

nstore::nstore(const nstore& o) :
  size_(o.size_),
  depth_(o.depth_)
{
  create();
  std::memcpy(buffer_, o.buffer_, size_ * depth_);
}

nstore::nstore(size_type n, unsigned d) :
  size_(n),
  depth_(d)
{
  if (d == 0)
    throw std::invalid_argument("depth may not be zero");
  if (d > sizeof(uint64_t))
    throw std::invalid_argument("depth > sizeof(uint64_t) is not supported");
  create();
}

nstore&
nstore::operator=(const nstore& o)
{
  if (size_ != o.size_ || depth_ != o.depth_) {
    destroy();
    size_ = o.size_;
    depth_ = o.depth_;
    create();
  }
  std::memcpy(buffer_, o.buffer_, size_ * depth_);
  return *this;
}

nstore&
nstore::operator+=(const nstore& o)
{
  if (size_ != o.size_)
    throw std::logic_error("sizes do not match");
  for (size_type i = 0; i < size_; ++i)
    write(i, read(i) + o.read(i));
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

uint64_t
nstore::read(size_type i)
  const
{
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_READ(T) \
    case sizeof(T): return ((T*)buffer_)[i]
    BOOST_HISTOGRAM_NSTORE_READ(uint8_t);
    BOOST_HISTOGRAM_NSTORE_READ(uint16_t);
    BOOST_HISTOGRAM_NSTORE_READ(uint32_t);
    BOOST_HISTOGRAM_NSTORE_READ(uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_READ
    default: assert(!"invalid depth");
  }
  return 0;
}

void
nstore::write(size_type i, uint64_t v)
{
  const uint64_t vmax = max_count();
  while (vmax < v) grow();

  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_WRITE(T) \
    case sizeof(T): {                       \
      ((T*)buffer_)[i] = v;                 \
      break;                                \
    }
    BOOST_HISTOGRAM_NSTORE_WRITE(uint8_t);
    BOOST_HISTOGRAM_NSTORE_WRITE(uint16_t);
    BOOST_HISTOGRAM_NSTORE_WRITE(uint32_t);
    BOOST_HISTOGRAM_NSTORE_WRITE(uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_WRITE
    default: assert(!"invalid depth");
  }
}

void
nstore::create()
{
  buffer_ = std::calloc(size_, depth_);
  if (!buffer_) throw std::bad_alloc();
}

void
nstore::destroy()
{
  std::free(buffer_); buffer_ = 0;
}

void
nstore::grow()
{
  if (depth_ == sizeof(uint64_t))
    throw std::overflow_error("depth > 64 bit is not supported");
  if (depth_ == 0 || buffer_ == 0)
    throw std::logic_error("cannot call grow on null buffer");
  buffer_ = std::realloc(buffer_, size_ * 2 * depth_);
  if (!buffer_) throw std::bad_alloc();
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_GROW(T0, T1)          \
    case sizeof(T0):                                     \
    for (size_type i = size_ - 1; i != size_type(-1); --i) \
      ((T1*)buffer_)[i] = ((T0*)buffer_)[i];             \
    break
    BOOST_HISTOGRAM_NSTORE_GROW(uint8_t, uint16_t);
    BOOST_HISTOGRAM_NSTORE_GROW(uint16_t, uint32_t);
    BOOST_HISTOGRAM_NSTORE_GROW(uint32_t, uint64_t);
    #undef BOOST_HISTOGRAM_NSTORE_GROW
  }
  depth_ *= 2;
}

uint64_t
nstore::max_count()
  const
{
  switch (depth_) {
    #define BOOST_HISTOGRAM_NSTORE_CASE(T) \
    case sizeof(T): return std::numeric_limits<T>::max()
    BOOST_HISTOGRAM_NSTORE_CASE(uint8_t);
    BOOST_HISTOGRAM_NSTORE_CASE(uint16_t);
    BOOST_HISTOGRAM_NSTORE_CASE(uint32_t);
    BOOST_HISTOGRAM_NSTORE_CASE(uint64_t);
    default: assert(!"invalid depth");
  }
  return 0;
}

}
}
}
