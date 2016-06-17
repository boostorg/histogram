#ifndef _BOOST_HISTOGRAM_DETAIL_NSTORE_HPP_
#define _BOOST_HISTOGRAM_DETAIL_NSTORE_HPP_

#include <boost/histogram/detail/wtype.hpp>
#include <boost/histogram/detail/zero_suppression.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/array.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/move/move.hpp>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <new> // for bad:alloc

namespace boost {
namespace histogram {
namespace detail {

// we rely on boost to guarantee that boost::uintX_t
// has a size of exactly X bits, so we only check wtype
BOOST_STATIC_ASSERT(sizeof(wtype) >= (2 * sizeof(uint64_t)));

class nstore {
  BOOST_COPYABLE_AND_MOVABLE(nstore)
public:
  typedef uintptr_t size_type;

  enum depth_type {
    d1 = sizeof(uint8_t),
    d2 = sizeof(uint16_t),
    d4 = sizeof(uint32_t),
    d8 = sizeof(uint64_t),
    dw = sizeof(wtype)
  };

  nstore();
  nstore(size_type, depth_type d = d1);
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
        std::memcpy(buffer_, o.buffer_, size_ * static_cast<int>(depth_));
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
    depth_(d1),
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
      depth_ = d1;
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
      #define BOOST_HISTOGRAM_NSTORE_INC(D, T)  \
      case D: {                                 \
        T& b = ((T*)buffer_)[i];                \
        if (b == std::numeric_limits<T>::max()) \
          grow(); /* and fall to next case */   \
        else { ++b; break; }                    \
      }
      BOOST_HISTOGRAM_NSTORE_INC(d1, uint8_t);
      BOOST_HISTOGRAM_NSTORE_INC(d2, uint16_t);
      BOOST_HISTOGRAM_NSTORE_INC(d4, uint32_t);
      BOOST_HISTOGRAM_NSTORE_INC(d8, uint64_t);
      #undef BOOST_HISTOGRAM_NSTORE_INC
      case dw: {
        wtype& b = ((wtype*)buffer_)[i];
        b += 1.0;
      } break;
    }
  }

  inline void increase(size_type i, double w) {
    if (depth_ != dw)
      wconvert();
    ((wtype*)buffer_)[i] += w;
  }

  double value(size_type) const;
  double variance(size_type) const;

  const void* buffer() const { return buffer_; }
  unsigned depth() const { return unsigned(depth_); }

private:
  size_type size_;
  depth_type depth_;
  void* buffer_;

  void* create(void*);
  void destroy();
  void grow();
  void wconvert();

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
        #define BOOST_HISTOGRAM_NSTORE_SAVE(D, T) \
        case D: {                              \
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
        BOOST_HISTOGRAM_NSTORE_SAVE(d1, uint8_t);
        BOOST_HISTOGRAM_NSTORE_SAVE(d2, uint16_t);
        BOOST_HISTOGRAM_NSTORE_SAVE(d4, uint32_t);
        BOOST_HISTOGRAM_NSTORE_SAVE(d8, uint64_t);
        BOOST_HISTOGRAM_NSTORE_SAVE(dw, wtype);
        #undef BOOST_HISTOGRAM_NSTORE_SAVE
      }
    }

    if (Archive::is_loading::value) {
      bool is_zero_suppressed = false;
      ar & is_zero_suppressed;
      switch (depth_) {
        #define BOOST_HISTOGRAM_NSTORE_LOAD(D, T) \
        case D:                        \
        if (is_zero_suppressed) {              \
          std::vector<T> buf;                  \
          ar & buf;                            \
          zero_suppression_decode<T>((T*)buffer_, size_, buf); \
        } else {                               \
          ar & serialization::make_array((T*)buffer_, size_); \
        } break
        BOOST_HISTOGRAM_NSTORE_LOAD(d1, uint8_t);
        BOOST_HISTOGRAM_NSTORE_LOAD(d2, uint16_t);
        BOOST_HISTOGRAM_NSTORE_LOAD(d4, uint32_t);
        BOOST_HISTOGRAM_NSTORE_LOAD(d8, uint64_t);
        BOOST_HISTOGRAM_NSTORE_LOAD(dw, wtype);
        #undef BOOST_HISTOGRAM_NSTORE_LOAD
      }
    }
  }
};

}
}
}

#endif
