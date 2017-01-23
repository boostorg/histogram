// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DYNAMIC_STORAGE_HPP_
#define _BOOST_HISTOGRAM_DYNAMIC_STORAGE_HPP_

#include <boost/histogram/detail/weight.hpp>
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

  static_assert(std::is_pod<detail::weight_t>::value, "weight_t must be POD");

  template <typename T>
  void increase_impl(detail::buffer& buf, std::size_t i) {
    auto& b = buf.at<T>(i);
    if (b < std::numeric_limits<T>::max())
      ++b;
    else {
      buf.grow<T>();
      using U = detail::next_storage_type<T>;
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
      buf.grow<T>();
      add_impl<detail::next_storage_type<T>>(buf, i, o);
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
  using weight_t = detail::weight_t;

public:
  using value_t = double;
  using variance_t = double;

  explicit
  dynamic_storage(std::size_t s) :
    buffer_(s)
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
};

inline
void dynamic_storage::increase(std::size_t i)
{
  switch (buffer_.id()) {
    case 0: buffer_.initialize<uint8_t>(); // and fall through
    case 1: increase_impl<uint8_t> (buffer_, i); break;
    case 2: increase_impl<uint16_t>(buffer_, i); break;
    case 3: increase_impl<uint32_t>(buffer_, i); break;
    case 4: increase_impl<uint64_t>(buffer_, i); break;
    case 6: ++(buffer_.at<weight_t>(i)); break;
  }
}

inline
void dynamic_storage::increase(std::size_t i, double w)
{
  buffer_.wconvert();
  buffer_.at<weight_t>(i).add_weight(w);
}

inline
dynamic_storage& dynamic_storage::operator+=(const dynamic_storage& o)
{
  if (o.depth()) {
    if (o.buffer_.id() == 6) {
      buffer_.wconvert();
      for (std::size_t i = 0; i < size(); ++i)
        buffer_.at<weight_t>(i) += o.buffer_.at<weight_t>(i);
    }
    else {
      auto i = size();
      while (i--) {
        uint64_t n = 0;
        switch (o.buffer_.id()) {
          /* case 0 is already excluded by the initial if statement */
          case 1: n = o.buffer_.at<uint8_t> (i); break;
          case 2: n = o.buffer_.at<uint16_t>(i); break;
          case 3: n = o.buffer_.at<uint32_t>(i); break;
          case 4: n = o.buffer_.at<uint64_t>(i); break;
        }
        switch (buffer_.id()) {
          case 0: buffer_.initialize<uint8_t>(); // and fall through
          case 1: add_impl<uint8_t> (buffer_, i, n); break;
          case 2: add_impl<uint16_t>(buffer_, i, n); break;
          case 3: add_impl<uint32_t>(buffer_, i, n); break;
          case 4: add_impl<uint64_t>(buffer_, i, n); break;
          case 6: buffer_.at<weight_t>(i) += n; break;
        }
      }
    }
  }
  return *this;
}

inline
dynamic_storage::value_t dynamic_storage::value(std::size_t i) const
{
  switch (buffer_.id()) {
    case 0: break;
    case 1: return buffer_.at<uint8_t> (i);
    case 2: return buffer_.at<uint16_t>(i);
    case 3: return buffer_.at<uint32_t>(i);
    case 4: return buffer_.at<uint64_t>(i);
    case 6: return buffer_.at<weight_t>(i).w;
  }
  return 0.0;
}

inline
dynamic_storage::variance_t dynamic_storage::variance(std::size_t i) const
{
  switch (buffer_.id()) {
    case 0: break;
    case 1: return buffer_.at<uint8_t> (i);
    case 2: return buffer_.at<uint16_t>(i);
    case 3: return buffer_.at<uint32_t>(i);
    case 4: return buffer_.at<uint64_t>(i);
    case 6: return buffer_.at<weight_t>(i).w2;
  }
  return 0.0;
}

}
}

#endif
