// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DYNAMIC_STORAGE_HPP_
#define _BOOST_HISTOGRAM_DYNAMIC_STORAGE_HPP_

#include <boost/histogram/detail/wtype.hpp>
#include <boost/histogram/detail/buffer.hpp>
#include <boost/histogram/detail/mpl.hpp>
#include <boost/assert.hpp>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <algorithm>
#include <limits>

namespace boost {
namespace histogram {

namespace {
  template <typename T> struct next_storage_type;
  template <> struct next_storage_type<uint8_t>  { typedef uint16_t type; };
  template <> struct next_storage_type<uint16_t> { typedef uint32_t type; };
  template <> struct next_storage_type<uint32_t> { typedef uint64_t type; };

  static_assert(std::is_pod<detail::wtype>::value, "wtype must be POD");

  template <typename T,
            typename U = typename next_storage_type<T>::type>
  void grow_impl(detail::buffer& buf) {
    static_assert(sizeof(U) >= sizeof(T), "U must be as large or larger than T");
    buf.depth(sizeof(U));
    std::copy_backward(buf.begin<T>(), buf.end<T>(), buf.end<U>());
  }

  template <typename T>
  void increase_impl(detail::buffer& buf, std::size_t i) {
    auto& b = buf.at<T>(i);
    if (b < std::numeric_limits<T>::max())
      ++b;
    else {
      grow_impl<T>(buf);
      using U = typename next_storage_type<T>::type;
      auto& b = buf.at<U>(i);
      ++b;
    }
  }

  template <>
  void increase_impl<uint64_t>(detail::buffer& buf, std::size_t i) {
    auto& b = buf.at<uint64_t>(i);
    if (b < std::numeric_limits<uint64_t>::max())
      ++b;
    else
      throw std::overflow_error("histogram overflow");
  }

  template <typename T>
  void add_impl(detail::buffer& buf, std::size_t i, uint64_t o) {
    auto& b = buf.at<T>(i);
    if (static_cast<T>(std::numeric_limits<T>::max() - b) >= o)
      b += o;
    else {
      grow_impl<T>(buf);
      add_impl<typename next_storage_type<T>::type>(buf, i, o);
    }
  }

  template <>
  void add_impl<uint64_t>(detail::buffer& buf, std::size_t i, uint64_t o) {
    auto& b = buf.at<uint64_t>(i);
    if (static_cast<uint64_t>(std::numeric_limits<uint64_t>::max() - b) >= o)
      b += o;
    else
      throw std::overflow_error("histogram overflow");
  }
}

class dynamic_storage {
  using wtype = detail::wtype;

public:
  using value_t = double;
  using variance_t = double;

  explicit
  dynamic_storage(std::size_t s) :
    buffer_(s, 0)
  {}

  dynamic_storage() = default;
  dynamic_storage(const dynamic_storage&) = default;
  dynamic_storage(dynamic_storage&&) = default;
  dynamic_storage& operator=(const dynamic_storage&) = default;
  dynamic_storage& operator=(dynamic_storage&&) = default;

  template <typename T,
            template <typename> class Storage,
            typename = detail::is_standard_integral<T>>
  dynamic_storage(const Storage<T>& o) :
    buffer_(o)
  {}

  template <typename T,
            template <typename> class Storage,
            typename = detail::is_standard_integral<T>>
  dynamic_storage& operator=(const Storage<T>& o)
  {
    buffer_ = o;
    return *this;
  }

  template <typename T,
            template <typename> class Storage,
            typename = detail::is_standard_integral<T>>
  dynamic_storage(Storage<T>&& o) :
    buffer_(std::move(o))
  {}

  template <typename T,
            template <typename> class Storage,
            typename = detail::is_standard_integral<T>>
  dynamic_storage& operator=(Storage<T>&& o)
  {
    buffer_ = std::move(o);
    return *this;
  }

  std::size_t size() const { return buffer_.size(); }
  unsigned depth() const { return buffer_.depth(); }
  const void* data() const { return buffer_.data(); }
  void increase(std::size_t i);
  void increase(std::size_t i, double w);
  value_t value(std::size_t i) const;
  variance_t variance(std::size_t i) const;
  dynamic_storage& operator+=(const dynamic_storage&);

private:
  detail::buffer buffer_;

  template <class Archive>
  friend void serialize(Archive&, dynamic_storage&, unsigned);

  void wconvert();
};

inline
void dynamic_storage::increase(std::size_t i)
{
  switch (buffer_.depth()) {
    case 0: buffer_.depth(sizeof(uint8_t)); // fall through
    case sizeof(uint8_t) : increase_impl<uint8_t> (buffer_, i); break;
    case sizeof(uint16_t): increase_impl<uint16_t>(buffer_, i); break;
    case sizeof(uint32_t): increase_impl<uint32_t>(buffer_, i); break;
    case sizeof(uint64_t): increase_impl<uint64_t>(buffer_, i); break;
    case sizeof(wtype): ++(buffer_.at<wtype>(i)); break;
  }
}

inline
void dynamic_storage::increase(std::size_t i, double w)
{
  wconvert();
  buffer_.at<wtype>(i).add_weight(w);
}

inline
dynamic_storage& dynamic_storage::operator+=(const dynamic_storage& o)
{
  if (o.depth()) {
    if (o.depth() == sizeof(wtype)) {
      wconvert();
      for (std::size_t i = 0; i < size(); ++i)
        buffer_.at<wtype>(i) += o.buffer_.at<wtype>(i);
    }
    else {
      auto i = size();
      while (i--) {
        uint64_t n = 0;
        switch (o.depth()) {
          case 0: /* nothing to do */ break;
          case sizeof(uint8_t) : n = o.buffer_.at<uint8_t> (i); break;
          case sizeof(uint16_t): n = o.buffer_.at<uint16_t>(i); break;
          case sizeof(uint32_t): n = o.buffer_.at<uint32_t>(i); break;
          case sizeof(uint64_t): n = o.buffer_.at<uint64_t>(i); break;
        }
        switch (buffer_.depth()) {
          case 0: buffer_.depth(sizeof(uint8_t)); // and fall through
          case sizeof(uint8_t) : add_impl<uint8_t> (buffer_, i, n); break;
          case sizeof(uint16_t): add_impl<uint16_t>(buffer_, i, n); break;
          case sizeof(uint32_t): add_impl<uint32_t>(buffer_, i, n); break;
          case sizeof(uint64_t): add_impl<uint64_t>(buffer_, i, n); break;
          case sizeof(wtype): buffer_.at<wtype>(i) += n; break;
        }
      }
    }
  }
  return *this;
}

inline
dynamic_storage::value_t dynamic_storage::value(std::size_t i) const
{
  switch (buffer_.depth()) {
    case 0: break;
    case sizeof(uint8_t) : return buffer_.at<uint8_t> (i);
    case sizeof(uint16_t): return buffer_.at<uint16_t>(i);
    case sizeof(uint32_t): return buffer_.at<uint32_t>(i);
    case sizeof(uint64_t): return buffer_.at<uint64_t>(i);
    case sizeof(wtype): return buffer_.at<wtype>(i).w;
  }
  return 0.0;
}

inline
dynamic_storage::variance_t dynamic_storage::variance(std::size_t i) const
{
  switch (buffer_.depth()) {
    case 0: break;
    case sizeof(uint8_t) : return buffer_.at<uint8_t> (i);
    case sizeof(uint16_t): return buffer_.at<uint16_t>(i);
    case sizeof(uint32_t): return buffer_.at<uint32_t>(i);
    case sizeof(uint64_t): return buffer_.at<uint64_t>(i);
    case sizeof(wtype): return buffer_.at<wtype>(i).w2;
  }
  return 0.0;
}

inline
void dynamic_storage::wconvert()
{
  switch (buffer_.depth()) {
    case 0: buffer_.depth(sizeof(wtype)); break;
    case sizeof(uint8_t) : grow_impl<uint8_t, wtype> (buffer_); break;
    case sizeof(uint16_t): grow_impl<uint16_t, wtype>(buffer_); break;
    case sizeof(uint32_t): grow_impl<uint32_t, wtype>(buffer_); break;
    case sizeof(uint64_t): grow_impl<uint64_t, wtype>(buffer_); break;
    case sizeof(wtype): /* do nothing */ break;
  }
}

}
}

#endif
