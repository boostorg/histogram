#ifndef _BOOST_HISTOGRAM_DETAIL_NSTORE_HPP_
#define _BOOST_HISTOGRAM_DETAIL_NSTORE_HPP_

#include <boost/histogram/detail/wtype.hpp>
#include <boost/histogram/detail/zero_suppression.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/array.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/move/move.hpp>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <new> // for bad:alloc

namespace boost {
namespace histogram {
namespace detail {

class nstore {
  BOOST_COPYABLE_AND_MOVABLE(nstore)
public:
  typedef uintptr_t size_type;

  nstore();
  nstore(size_type, unsigned d = sizeof(uint8_t));
  ~nstore() { destroy(); }

  // copy semantics
  nstore(const nstore& o) :
    size_(o.size_),
    depth_(o.depth_),
    buffer_(create(o.buffer_))
  {}

  nstore& operator=(BOOST_COPY_ASSIGN_REF(nstore) o)
  {
    if (this != &o) {
      if (size_ == o.size_ && depth_ == o.depth_) {
        std::memcpy(buffer_, o.buffer_, size_ * depth_);
      } else {
        destroy();
        size_ = o.size_;
        depth_ = o.depth_;
        buffer_ = create(o.buffer_);
      }
    }
    return *this;
  }

  // move semantics
  nstore(BOOST_RV_REF(nstore) o) :
    size_(0),
    depth_(0),
    buffer_(0)
  {
    std::swap(size_, o.size_);
    std::swap(depth_, o.depth_);
    std::swap(buffer_, o.buffer_);
  }

  nstore& operator=(BOOST_RV_REF(nstore) o)
  {
    if (this != &o) {
      size_ = 0;
      depth_ = 0;
      destroy();
      std::swap(size_, o.size_);
      std::swap(depth_, o.depth_);
      std::swap(buffer_, o.buffer_);
    }
    return *this;
  }

  nstore& operator+=(const nstore&);
  bool operator==(const nstore&) const;

  inline void increase(size_type i) {
    switch (depth_) {
      case sizeof(wtype): {
        wtype& b = ((wtype*)buffer_)[i];
        b += 1.0;        
      } break;
      #define BOOST_HISTOGRAM_NSTORE_INC(T)     \
      case sizeof(T): {                         \
        T& b = ((T*)buffer_)[i];                \
        if (b == std::numeric_limits<T>::max()) \
          grow(); /* and fall to next case */   \
        else { ++b; break; }                    \
      }
      case 0: buffer_ = create(0);
      BOOST_HISTOGRAM_NSTORE_INC(uint8_t);
      BOOST_HISTOGRAM_NSTORE_INC(uint16_t);
      BOOST_HISTOGRAM_NSTORE_INC(uint32_t);
      BOOST_HISTOGRAM_NSTORE_INC(uint64_t);
      #undef BOOST_HISTOGRAM_NSTORE_INC
      default: increase(i, 1.0);
    }
  }

  inline void increase(size_type i, double w) {
    if (depth_ != sizeof(wtype))
      wconvert();
    ((wtype*)buffer_)[i] += w;
  }

  double value(size_type) const;
  double variance(size_type) const;

  const void* buffer() const { return buffer_; }
  unsigned depth() const { return depth_; }

private:
  size_type size_;
  unsigned depth_;
  void* buffer_;

  void* create(void*);
  void destroy();
  void grow();
  void wconvert();

  uint64_t max_count() const
  {
    switch (depth_) {
      #define BOOST_HISTOGRAM_NSTORE_CASE(T) \
      case sizeof(T): return std::numeric_limits<T>::max()
      BOOST_HISTOGRAM_NSTORE_CASE(uint8_t);
      BOOST_HISTOGRAM_NSTORE_CASE(uint16_t);
      BOOST_HISTOGRAM_NSTORE_CASE(uint32_t);
      BOOST_HISTOGRAM_NSTORE_CASE(uint64_t);
      #undef BOOST_HISTOGRAM_NSTORE_CASE
      default: BOOST_ASSERT(!"invalid depth");
    }
    return 0;
  }

  uint64_t ivalue(size_type) const;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    const size_type s = size_;
    const unsigned d = depth_;
    ar & size_;
    ar & depth_;
    if (s != size_ || d != depth_) {
      // realloc is safe if buffer_ is null
      buffer_ = std::realloc(buffer_, size_ * depth_);
    }
    if (buffer_ == 0 && size_ > 0)
      throw std::bad_alloc();

    if (Archive::is_saving::value) {
      switch (depth_) {
        #define BOOST_HISTOGRAM_NSTORE_SAVE(T) \
        case sizeof(T): {                      \
          std::vector<T> buf;                  \
          if (zero_suppression_encode<T>(buf, (T*)buffer_, size_)) { \
            bool is_zero_suppressed = true;    \
            ar & is_zero_suppressed;           \
            ar & buf;                          \
          } else {                             \
            bool is_zero_suppressed = false;   \
            ar & is_zero_suppressed;           \
            ar & serialization::make_array((T*)buffer_, size_); \
          }                                    \
        } break
        BOOST_HISTOGRAM_NSTORE_SAVE(uint8_t);
        BOOST_HISTOGRAM_NSTORE_SAVE(uint16_t);
        BOOST_HISTOGRAM_NSTORE_SAVE(uint32_t);
        BOOST_HISTOGRAM_NSTORE_SAVE(uint64_t);
        BOOST_HISTOGRAM_NSTORE_SAVE(wtype);
        #undef BOOST_HISTOGRAM_NSTORE_SAVE
        default: BOOST_ASSERT(!"invalid depth");
      }
    }

    if (Archive::is_loading::value) {
      bool is_zero_suppressed = false;
      ar & is_zero_suppressed;
      switch (depth_) {
        #define BOOST_HISTOGRAM_NSTORE_LOAD(T) \
        case sizeof(T):                        \
        if (is_zero_suppressed) {              \
          std::vector<T> buf;                  \
          ar & buf;                            \
          zero_suppression_decode<T>((T*)buffer_, size_, buf); \
        } else {                               \
          ar & serialization::make_array((T*)buffer_, size_); \
        } break
        BOOST_HISTOGRAM_NSTORE_LOAD(uint8_t);
        BOOST_HISTOGRAM_NSTORE_LOAD(uint16_t);
        BOOST_HISTOGRAM_NSTORE_LOAD(uint32_t);
        BOOST_HISTOGRAM_NSTORE_LOAD(uint64_t);
        BOOST_HISTOGRAM_NSTORE_LOAD(wtype);
        #undef BOOST_HISTOGRAM_NSTORE_LOAD
        default: BOOST_ASSERT(!"invalid depth");
      }
    }
  }
};

}
}
}

#endif
