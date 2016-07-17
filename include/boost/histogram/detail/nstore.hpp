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
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>
#include <new> // for bad_alloc

namespace boost {
namespace histogram {
namespace detail {

// we rely on boost to guarantee that boost::uintX_t
// has a size of exactly X bits, so we only check wtype
BOOST_STATIC_ASSERT(sizeof(wtype) >= (2 * sizeof(uint64_t)));

template <typename T> struct next_storage_type;
template <> struct next_storage_type<uint8_t>  { typedef uint16_t type; };
template <> struct next_storage_type<uint16_t> { typedef uint32_t type; };
template <> struct next_storage_type<uint32_t> { typedef uint64_t type; };
template <> struct next_storage_type<uint64_t> { typedef wtype type; };

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

  inline
  void increase(size_type i) {
    switch (depth_) {
      case d1: increase_impl<uint8_t> (i); break;
      case d2: increase_impl<uint16_t>(i); break;
      case d4: increase_impl<uint32_t>(i); break;
      case d8: increase_impl<uint64_t>(i); break;
      case dw: increase_impl<wtype>(i); break;
    }
  }

  inline
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
  inline
  typename disable_if<is_same<T, wtype>, void>::type
  increase_impl(size_type i)
  {
    T& b = static_cast<T*>(buffer_)[i];
    if (b == std::numeric_limits<T>::max())
    {
      grow_impl<T>();
      typedef typename  next_storage_type<T>::type U;
      U& b = static_cast<U*>(buffer_)[i];
      ++b;
    }
    else {
      ++b;
    }
  }

  template<class T>
  inline
  typename enable_if<is_same<T, wtype>, void>::type
  increase_impl(size_type i)
  {
    ++(static_cast<wtype*>(buffer_)[i]);
  }

  template<class T>
  typename disable_if<is_same<T, wtype>, void>::type
  add_impl(size_type i, const uint64_t & oi)
  {
    T& b = static_cast<T*>(buffer_)[i];
    if (static_cast<T>(std::numeric_limits<T>::max() - b) >= oi) {
      b += oi;
    } else {
      grow_impl<T>();
      add_impl<typename next_storage_type<T>::type>(i, oi);
    }
  }

  template<class T>
  typename enable_if<is_same<T, wtype>, void>::type
  add_impl(size_type i, const uint64_t & oi)
  {
    static_cast<wtype*>(buffer_)[i] += wtype(oi);
  }

  template<typename T>
  void grow_impl()
  {
    BOOST_ASSERT(size_ > 0);

    typedef typename next_storage_type<T>::type U;
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
  friend void serialize_save_impl(Archive & ar, const nstore &,
                                  unsigned version);
  template<class T, class Archive>
  friend void serialize_load_impl(Archive & ar, nstore &,
                                  bool is_zero_suppressed, unsigned version);
  template <class Archive>
  friend void serialize(Archive& ar, nstore &, unsigned version);
};


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
      static_cast<wtype*>(buffer_)[i] += static_cast<wtype*>(o.buffer_)[i];
  }
  else {
    size_type i = size_;
    while (i--) {
      const uint64_t oi = o.ivalue(i);
      switch (depth_) {
        case d1: add_impl<uint8_t> (i, oi); break;
        case d2: add_impl<uint16_t> (i, oi); break;
        case d4: add_impl<uint32_t> (i, oi); break;
        case d8: add_impl<uint64_t> (i, oi); break;
        case dw: add_impl<wtype> (i, oi); break;
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
    case d1: return static_cast<uint8_t *>(buffer_)[i];
    case d2: return static_cast<uint16_t*>(buffer_)[i];
    case d4: return static_cast<uint32_t*>(buffer_)[i];
    case d8: return static_cast<uint64_t*>(buffer_)[i];
    case dw: return static_cast<wtype*>(buffer_)[i].w2;
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
  switch (depth_) {
    case d1: grow_impl<uint8_t>(); break;
    case d2: grow_impl<uint16_t>(); break;
    case d4: grow_impl<uint32_t>(); break;
    case d8: grow_impl<uint64_t>(); break;
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
  switch (depth_) {
    case d1: wconvert_impl<uint8_t> (); break;
    case d2: wconvert_impl<uint16_t>(); break;
    case d4: wconvert_impl<uint32_t>(); break;
    case d8: wconvert_impl<uint64_t>(); break;
    case dw: BOOST_ASSERT(!"never arrive here");
  }
  depth_ = dw;
}

uint64_t
nstore::ivalue(size_type i)
  const
{
  switch (depth_) {
    case d1: return static_cast<uint8_t *>(buffer_)[i];
    case d2: return static_cast<uint16_t*>(buffer_)[i];
    case d4: return static_cast<uint32_t*>(buffer_)[i];
    case d8: return static_cast<uint64_t*>(buffer_)[i];
    case dw: BOOST_ASSERT(!"never arrive here");
  }
  return 0;
}

}
}
}

#endif
