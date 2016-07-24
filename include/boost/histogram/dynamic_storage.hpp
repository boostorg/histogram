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
    dynamic_storage(o.size())
  {
    for (std::size_t i = 0, n = size(); i < n; ++i) {
      switch (depth_) {
        case sizeof(uint8_t): add_impl<uint8_t, T>(i, o.data_.template get<T>(i)); break;
        case sizeof(uint16_t): add_impl<uint16_t, T>(i, o.data_.template get<T>(i)); break;
        case sizeof(uint32_t): add_impl<uint32_t, T>(i, o.data_.template get<T>(i)); break;
        case sizeof(uint64_t): add_impl<uint64_t, T>(i, o.data_.template get<T>(i)); break;
        case sizeof(wtype): add_impl<wtype, T>(i, o.data_.template get<T>(i)); break;
      }
    }
  }

  template <typename T,
            typename std::enable_if<
              (std::is_integral<T>::value &&
               (sizeof(T) & (sizeof(T) - 1)) == 0 && // size in 1,2,4,8
               sizeof(T) < sizeof(uint64_t)),
              int
            >::type = 0
  >
  dynamic_storage(static_storage<T>&& o) :
    data_(std::move(o.data_)),
    depth_(sizeof(T))
  {}

  template <typename T>
  dynamic_storage& operator=(const static_storage<T>& o)
  {
    *this = dynamic_storage(o); // leverage copy ctor, clang says: don't move
    return *this;
  }

  template <typename T>
  typename std::enable_if<
    (std::is_integral<T>::value &&
     (sizeof(T) & (sizeof(T) - 1)) == 0 && // size in 1,2,4,8
     sizeof(T) < sizeof(uint64_t)),
    dynamic_storage&
  >::type
  operator=(static_storage<T>&& o)
  {
    data_ = std::move(o.data_);
    depth_ = sizeof(T);
    return *this;
  }

  std::size_t size() const { return data_.nbytes() / depth_; }
  unsigned depth() const { return depth_; }
  const void* data() const { return &data_.get<char>(0); }
  void increase(std::size_t i);
  void increase(std::size_t i, double w);
  value_t value(std::size_t i) const;
  variance_t variance(std::size_t i) const;
  dynamic_storage& operator+=(const dynamic_storage&);

private:
  buffer_t data_;
  unsigned depth_ = sizeof(uint8_t);

  void wconvert();

  template <typename T>
  typename std::enable_if<!std::is_same<T, wtype>::value, void>::type
  increase_impl(std::size_t i)
  {
    auto& b = data_.get<T>(i);
    if (b < std::numeric_limits<T>::max())
      ++b;
    else {
      grow_impl<T>();
      using U = typename next_storage_type<T>::type;
      auto& b = data_.get<U>(i);
      ++b;
    }
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, wtype>::value, void>::type
  increase_impl(std::size_t i)
  {
    ++(data_.get<wtype>(i));
  }

  template <typename T, typename O = uint64_t>
  typename std::enable_if<!std::is_same<T, wtype>::value, void>::type
  add_impl(std::size_t i, O o)
  {
    auto& b = data_.get<T>(i);
    if ((std::numeric_limits<T>::max() - b) >= o) {
      b += o;
    } else {
      grow_impl<T>();
      add_impl<typename next_storage_type<T>::type, O>(i, o);
    }
  }

  template <typename T, typename O = uint64_t>
  typename std::enable_if<std::is_same<T, wtype>::value, void>::type
  add_impl(std::size_t i, O o)
  {
    data_.get<wtype>(i) += o;
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
  data_.get<wtype>(i).add_weight(w);
}

dynamic_storage& dynamic_storage::operator+=(const dynamic_storage& o)
{
  BOOST_ASSERT(size() == o.size());

  if (o.depth_ == sizeof(wtype)) {
    if (depth_ != sizeof(wtype))
      wconvert();
    for (std::size_t i = 0, n = size(); i < n; ++i)
      data_.get<wtype>(i) += o.data_.get<wtype>(i);
  }
  else {
    std::size_t i = size();
    while (i--) {
      uint64_t n = 0;
      switch (o.depth_) {
        case sizeof(uint8_t): n = o.data_.get<uint8_t>(i); break;
        case sizeof(uint16_t): n = o.data_.get<uint16_t>(i); break;
        case sizeof(uint32_t): n = o.data_.get<uint32_t>(i); break;
        case sizeof(uint64_t): n = o.data_.get<uint64_t>(i); break;
        default: BOOST_ASSERT(!"never arrive here");
      }
      switch (depth_) {
        case sizeof(uint8_t): add_impl<uint8_t>(i, n); break;
        case sizeof(uint16_t): add_impl<uint16_t>(i, n); break;
        case sizeof(uint32_t): add_impl<uint32_t>(i, n); break;
        case sizeof(uint64_t): add_impl<uint64_t>(i, n); break;
        case sizeof(wtype): add_impl<wtype>(i, n); break;
        default: BOOST_ASSERT(!"never arrive here");
      }
    }
  }
  return *this;
}

dynamic_storage::value_t dynamic_storage::value(std::size_t i) const
{
  switch (depth_) {
    case sizeof(uint8_t): return data_.get<uint8_t>(i);
    case sizeof(uint16_t): return data_.get<uint16_t>(i);
    case sizeof(uint32_t): return data_.get<uint32_t>(i);
    case sizeof(uint64_t): return data_.get<uint64_t>(i);
    case sizeof(wtype): return data_.get<wtype>(i).w;
    default: BOOST_ASSERT(!"never arrive here");
  }
  return 0.0;
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

}
}

#endif
