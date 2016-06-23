// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

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
