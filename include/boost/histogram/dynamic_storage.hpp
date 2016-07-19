// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DYNAMIC_STORAGE_HPP_
#define _BOOST_HISTOGRAM_DYNAMIC_STORAGE_HPP_

#include <boost/histogram/detail/wtype.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/static_storage.hpp>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <limits>
#include <algorithm>

namespace boost {
namespace histogram {

namespace {
  template <typename T> struct next_storage_type;
  template <> struct next_storage_type<uint8_t>  { typedef uint16_t type; };
  template <> struct next_storage_type<uint16_t> { typedef uint32_t type; };
  template <> struct next_storage_type<uint32_t> { typedef uint64_t type; };
  template <> struct next_storage_type<uint64_t> { typedef detail::wtype type; };

  template <unsigned Depth> struct depth_to_type;
  template <> struct depth_to_type<sizeof(uint8_t)> { typedef uint8_t type; };
  template <> struct depth_to_type<sizeof(uint16_t)> { typedef uint16_t type; };
  template <> struct depth_to_type<sizeof(uint32_t)> { typedef uint32_t type; };
  template <> struct depth_to_type<sizeof(uint64_t)> { typedef uint64_t type; };

  // we rely on C++11 to guarantee that uintX_t has a size of exactly X bits,
  // so we only check size of wtype
  static_assert((sizeof(detail::wtype) >= (2 * sizeof(uint64_t))),
                "wtype is too narrow");
  static_assert(std::is_pod<detail::wtype>::value, "wtype must be POD");
}

class dynamic_storage {
  using wtype = detail::wtype;
  using buffer_t = detail::buffer_t;

public:
  using value_t = double ;
  using variance_t = double;

  dynamic_storage(std::size_t n = 0) :
    data_(n * sizeof(uint8_t)),
    depth_(sizeof(uint8_t))
  {}

  dynamic_storage(const dynamic_storage&) = default;
  dynamic_storage(dynamic_storage&&) = default;
  dynamic_storage& operator=(const dynamic_storage&) = default;
  dynamic_storage& operator=(dynamic_storage&&) = default;

  template <typename T>
  dynamic_storage(const static_storage<T>& o) :
    data_(o.data_),
    depth_(sizeof(T))
  {
    static_assert(std::is_integral<T>::value,
                  "storage type of source must be an integer");
    static_assert((sizeof(T) & (sizeof(T) - 1)) == 0,
                  "source depth of storage type has to be power of 2");
    static_assert(sizeof(T) < sizeof(uint64_t),
                  "source depth of storage type must be less than 64 bits");
  }

  template <typename T>
  dynamic_storage(static_storage<T>&& o) :
    data_(std::move(o.data_)),
    depth_(sizeof(T))
  {
    static_assert(std::is_integral<T>::value,
                  "storage type of source must be an integer");
    static_assert((sizeof(T) & (sizeof(T) - 1)) == 0,
                  "source depth of storage type has to be power of 2");
    static_assert(sizeof(T) < sizeof(uint64_t),
                  "source depth of storage type must be less than 64 bits");
  }

  template <typename T>
  dynamic_storage& operator=(const static_storage<T>& o)
  {
    static_assert(std::is_integral<T>::value,
                  "storage type is required to be an integer");
    static_assert((sizeof(T) & (sizeof(T) - 1)) == 0,
                  "depth of storage type has to be power of 2");
    static_assert(sizeof(T) < sizeof(uint64_t),
                  "depth of storage type must be less than 64 bits");
    data_ = o.data_;
    depth_ = sizeof(T);
    return *this;
  }

  template <typename T>
  dynamic_storage& operator=(static_storage<T>&& o)
  {
    static_assert(std::is_integral<T>::value,
                  "storage type is required to be an integer");
    static_assert((sizeof(T) & (sizeof(T) - 1)) == 0,
                  "depth of storage type has to be power of 2");
    static_assert(sizeof(T) < sizeof(uint64_t),
                  "depth of storage type must be less than 64 bits");
    data_ = std::move(o.data_);
    depth_ = sizeof(T);
    return *this;
  }

  std::size_t size() const { return data_.nbytes() / depth_; }
  unsigned depth() const { return depth_; }
  void increase(std::size_t i);
  void increase(std::size_t i, double w);
  value_t value(std::size_t i) const;
  variance_t variance(std::size_t i) const;
  bool operator==(const dynamic_storage&) const;
  dynamic_storage& operator+=(const dynamic_storage&);

private:
  buffer_t data_;
  unsigned depth_ = sizeof(uint8_t);

  template <typename T>
  typename std::enable_if<!std::is_same<T, wtype>::value, void>::type
  increase_impl(std::size_t i)
  {
    auto& b = data_.get<T>(i);
    if (b == std::numeric_limits<T>::max()) {
      grow_impl<T>();
      using U = typename next_storage_type<T>::type;
      auto& b = data_.get<U>(i);
      ++b;
    }
    else {
      ++b;
    }
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, wtype>::value, void>::type
  increase_impl(std::size_t i)
  {
    ++(data_.get<wtype>(i));
  }

  template <typename T>
  typename std::enable_if<!std::is_same<T, wtype>::value, void>::type
  add_impl(std::size_t i, uint64_t oi)
  {
    auto& b = data_.get<T>(i);
    if ((std::numeric_limits<T>::max() - b) >= oi) {
      b += oi;
    } else {
      grow_impl<T>();
      add_impl<typename next_storage_type<T>::type>(i, oi);
    }
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, wtype>::value, void>::type
  add_impl(std::size_t i, uint64_t oi)
  {
    data_.get<wtype>(i) += oi;
  }

  template <typename T>
  void grow_impl()
  {
    using U = typename next_storage_type<T>::type;
    const auto n = size();
    depth_ = sizeof(U);
    data_.resize(n * depth_);
    auto buf_in = &(data_.get<T>(0));
    auto buf_out = &(data_.get<U>(0));
    std::copy_backward(buf_in, buf_in + n, buf_out + n);
   }

  template <typename T>
  void wconvert_copy_impl(std::size_t n)
  {
    auto buf_in = &(data_.get<T>(0));
    auto buf_out = &(data_.get<wtype>(0));
    std::copy_backward(buf_in, buf_in + n, buf_out + n);
  }

  void grow();
  void wconvert();

  uint64_t ivalue(std::size_t) const;
};

void dynamic_storage::increase(std::size_t i)
{
  switch (depth_) {
    case sizeof(uint8_t): increase_impl<uint8_t> (i); break;
    case sizeof(uint16_t): increase_impl<uint16_t>(i); break;
    case sizeof(uint32_t): increase_impl<uint32_t>(i); break;
    case sizeof(uint64_t): increase_impl<uint64_t>(i); break;
    case sizeof(wtype): increase_impl<wtype>(i); break;
    default: BOOST_ASSERT(!"never arrive here");
  }
}

void dynamic_storage::increase(std::size_t i, double w)
{
  if (depth_ != sizeof(wtype))
    wconvert();
  data_.get<wtype>(i) += w;
}

dynamic_storage& dynamic_storage::operator+=(const dynamic_storage& o)
{
  // make depth of lhs as large as rhs
  if (depth_ < o.depth_) {
    if (o.depth_ == sizeof(wtype))
      wconvert();
    else
      while (depth_ < o.depth_)
        grow();
  }

  // now add the content of lhs
  if (depth_ == sizeof(wtype)) {
    for (std::size_t i = 0, n = size(); i < n; ++i)
      data_.get<wtype>(i) += o.data_.get<wtype>(i);
  }
  else {
    std::size_t i = size();
    while (i--) {
      const uint64_t oi = o.ivalue(i);
      switch (depth_) {
        case sizeof(uint8_t): add_impl<uint8_t>(i, oi); break;
        case sizeof(uint16_t): add_impl<uint16_t>(i, oi); break;
        case sizeof(uint32_t): add_impl<uint32_t>(i, oi); break;
        case sizeof(uint64_t): add_impl<uint64_t>(i, oi); break;
        case sizeof(wtype): add_impl<wtype>(i, oi); break;
        default: BOOST_ASSERT(!"never arrive here");
      }
    }
  }
  return *this;
}

bool dynamic_storage::operator==(const dynamic_storage& o) const
{
  return depth_ == o.depth_ && data_ == o.data_;
}

dynamic_storage::value_t dynamic_storage::value(std::size_t i) const
{
  if (depth_ < sizeof(wtype))
    return ivalue(i);
  return data_.get<wtype>(i).w;
}

dynamic_storage::variance_t dynamic_storage::variance(std::size_t i) const
{
  switch (depth_) {
    case sizeof(uint8_t): return data_.get<uint8_t>(i);
    case sizeof(uint16_t): return data_.get<uint16_t>(i);
    case sizeof(uint32_t): return data_.get<uint32_t>(i);
    case sizeof(uint64_t): return data_.get<uint64_t>(i);
    case sizeof(wtype): return data_.get<wtype>(i).w2;
    default: BOOST_ASSERT(!"never arrive here");
  }
  return 0.0;
}

void dynamic_storage::grow()
{
  switch (depth_) {
    case sizeof(uint8_t): grow_impl<uint8_t>(); break;
    case sizeof(uint16_t): grow_impl<uint16_t>(); break;
    case sizeof(uint32_t): grow_impl<uint32_t>(); break;
    case sizeof(uint64_t): grow_impl<uint64_t>(); break;
    default: BOOST_ASSERT(!"never arrive here");
  }
}

void dynamic_storage::wconvert()
{
  BOOST_ASSERT(depth_ < sizeof(wtype));
  const auto n = size();
  const auto d = depth_;
  depth_ = sizeof(wtype);
  data_.resize(n * depth_);
  switch (d) {
    case sizeof(uint8_t): wconvert_copy_impl<uint8_t> (n); break;
    case sizeof(uint16_t): wconvert_copy_impl<uint16_t>(n); break;
    case sizeof(uint32_t): wconvert_copy_impl<uint32_t>(n); break;
    case sizeof(uint64_t): wconvert_copy_impl<uint64_t>(n); break;
    default: BOOST_ASSERT(!"never arrive here");
  }
}

uint64_t dynamic_storage::ivalue(std::size_t i)
  const
{
  switch (depth_) {
    case sizeof(uint8_t): return data_.get<uint8_t>(i);
    case sizeof(uint16_t): return data_.get<uint16_t>(i);
    case sizeof(uint32_t): return data_.get<uint32_t>(i);
    case sizeof(uint64_t): return data_.get<uint64_t>(i);
    default: BOOST_ASSERT(!"never arrive here");
  }
  return 0;
}

}
}

#endif
