// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_NSTORE_HPP_
#define _BOOST_HISTOGRAM_DETAIL_NSTORE_HPP_

#include <boost/histogram/detail/wtype.hpp>
#include <boost/histogram/detail/zero_suppression.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/move/move.hpp>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>
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
    size_(o.size_),
    depth_(o.depth_),
    buffer_(o.buffer_)
  {
	o.depth_ = d1;
	o.size_ = 0;
	o.buffer_ = static_cast<void*>(0);
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

  void increase(size_type i) {
    switch (depth_) {
      case d1: if (increase_impl_<uint8_t> (i)) break;
      case d2: if (increase_impl_<uint16_t>(i)) break;
      case d4: if (increase_impl_<uint32_t>(i)) break;
      case d8: if (increase_impl_<uint64_t>(i)) break;
      case dw: {
        wtype& b = static_cast<wtype*>(buffer_)[i];
        b += 1.0;
      } break;
    }
  }

  void increase(size_type i, double w) {
    if (depth_ != dw)
      wconvert();
    static_cast<wtype*>(buffer_)[i] += w;
  }

  double value(size_type) const;
  double variance(size_type) const;

  const void* buffer() const { return buffer_; }
  unsigned depth() const { return unsigned(depth_); }

private:

  template<class T>
  bool increase_impl_(size_type i)
  {
      T& b = static_cast<T*>(buffer_)[i];
      if (b == std::numeric_limits<T>::max())
      {
        grow(); /* and fall to next case */
        return false;
      }
      else {
    	  ++b;
    	  return true;
      }
  }

  template<typename T>
  bool add_impl_(size_type i, const uint64_t & oi)
  {
    T& b = ((T*)buffer_)[i];
    if (T(std::numeric_limits<T>::max() - b) >= oi) {
  	b += oi;
  	return true;
    } else grow(); /* and fall through */
    return false;
  }

  template<typename T, typename U>
  void grow_impl()
  {
      depth_ = static_cast<depth_type>(sizeof(U));
      buffer_ = std::realloc(buffer_, size_ * static_cast<int>(depth_));
      if (!buffer_) throw std::bad_alloc();


      T* buf_in = static_cast<T*>(buffer_);
      U* buf_out = static_cast<U*>(buffer_);

      std::copy_backward(buf_in, buf_in + size_, buf_out + size_);
   }

  template<typename T>
  void wconvert_impl()
  {
      T*     buf_in = static_cast<T*>(buffer_);
      wtype* buf_out = static_cast<wtype*>(buffer_);
      std::copy_backward(buf_in, buf_in + size_, buf_out + size_);

  }

  size_type size_;
  depth_type depth_;
  void* buffer_;

  void* create(void*);
  void destroy();
  void grow();
  void wconvert();

  uint64_t ivalue(size_type) const;

  template<class T, class Archive>
  friend void serialize_save_impl(Archive & ar, const nstore &, unsigned version);
  template<class T, class Archive>
  friend void serialize_load_impl(Archive & ar, nstore &,
		                          bool is_zero_suppressed, unsigned version);
  template <class Archive>
  friend void serialize(Archive& ar, nstore &, unsigned version);
};

}
}
}

#endif
