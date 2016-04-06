#ifndef _BOOST_HISTOGRAM_NSTORE_HPP_
#define _BOOST_HISTOGRAM_NSTORE_HPP_

#include <boost/serialization/access.hpp>
#include <boost/serialization/array.hpp>
#include <boost/cstdint.hpp>
#include <boost/histogram/detail/zero_suppression.hpp>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <new> // for bad:alloc

namespace boost {
namespace histogram {

class nstore {
public:
  typedef uint64_t size_type;
  typedef unsigned depth_type;

public:
  nstore();
  nstore(const nstore&);
  nstore(size_type, depth_type d = sizeof(uint8_t));
  ~nstore() { destroy(); }

  nstore& operator=(const nstore&);
  nstore& operator+=(const nstore&);
  bool operator==(const nstore&) const;

  uint64_t read(size_type) const;
  void write(size_type, uint64_t);

  inline void increase(size_type i) {
    switch (depth_) {
      #define BOOST_HISTOGRAM_NSTORE_INC(T)     \
      case sizeof(T): {                         \
        T& b = ((T*)buffer_)[i];                \
        if (b == std::numeric_limits<T>::max()) \
          grow(); /* and fall to next case */   \
        else { ++b; break; }                    \
      }
      BOOST_HISTOGRAM_NSTORE_INC(uint8_t);
      BOOST_HISTOGRAM_NSTORE_INC(uint16_t);
      BOOST_HISTOGRAM_NSTORE_INC(uint32_t);
      BOOST_HISTOGRAM_NSTORE_INC(uint64_t);
      #undef BOOST_HISTOGRAM_NSTORE_INC
      default: assert(!"invalid depth");
    }
  }

  const void* buffer() const { return buffer_; }
  depth_type depth() const { return depth_; }

private:
  size_type size_;
  depth_type depth_;
  void* buffer_;

  void create();
  void destroy();
  void grow();

  uint64_t max_count() const;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    const size_type s = size_;
    const depth_type d = depth_;
    ar & size_;
    ar & depth_;
    if (s != size_ || d != depth_) {
      // realloc is safe if buffer_ is null
      buffer_ = std::realloc(buffer_, size_ * depth_);
    }
    if (buffer_ == 0 && size_ > 0)
      throw std::bad_alloc();

    if (Archive::is_saving::value) {
      std::vector<char> buf;
      if (detail::zero_suppression_encode(buf, size_ * depth_,
                                          (char*)buffer_, size_ * depth_)) {
        bool is_zero_suppressed = true;
        ar & is_zero_suppressed;
        ar & buf;
      } else {
        bool is_zero_suppressed = false;
        ar & is_zero_suppressed;
        ar & serialization::make_array((char*)buffer_, size_ * depth_);
      }
    } else {
      bool is_zero_suppressed = false;
      ar & is_zero_suppressed;
      if (is_zero_suppressed) {
        std::vector<char> buf;
        ar & buf;
        detail::zero_suppression_decode((char*)buffer_, size_ * depth_, buf);
      } else {
        ar & serialization::make_array((char*)buffer_, size_ * depth_);
      }
    }
  }
};

}
}

#endif
