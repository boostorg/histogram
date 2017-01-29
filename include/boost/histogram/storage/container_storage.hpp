// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_CONTAINER_HPP_
#define _BOOST_HISTOGRAM_STORAGE_CONTAINER_HPP_

#include <algorithm>
#include <cstddef>

namespace boost {
namespace histogram {

namespace detail {
  template <typename Container>
  void init(Container& c, unsigned s) {
    if (c.size() != s) {
      c = Container(s, typename Container::value_type(0));
    } else {
      std::fill(c.begin(), c.end(), typename Container::value_type(0));      
    }
  }

  // template signature of both std::array and boost::array
  template <template <class, std::size_t> class Array, typename T, std::size_t N>
  void init(Array<T, N>& c, unsigned) {
    std::fill(c.begin(), c.end(), typename Array<T, N>::value_type(0));
  }
}

template <typename Container>
class container_storage {
public:
  using value_type = typename Container::value_type;

  container_storage() : c_() {}

  explicit container_storage(std::size_t s)
  {
    detail::init(c_, s);
  }

  container_storage(const container_storage&) = default;
  container_storage& operator=(const container_storage&) = default;
  container_storage(container_storage&&) = default;
  container_storage& operator=(container_storage&&) = default;

  template <typename OtherStorage,
            typename = detail::is_storage<OtherStorage>>
  container_storage(const OtherStorage& other)
  {
    detail::init(c_, other.size());
    for (std::size_t i = 0; i < c_.size(); ++i)
      c_[i] = other.value(i);
  }

  template <typename OtherStorage,
            typename = detail::is_storage<OtherStorage>>
  container_storage& operator=(const OtherStorage& other)
  {
    detail::init(c_, other.size());
    for (std::size_t i = 0; i < c_.size(); ++i)
      c_[i] = other.value(i);
    return *this;
  }

  std::size_t size() const { return c_.size(); }
  void increase(std::size_t i) { ++(c_[i]); }
  value_type value(std::size_t i) const { return c_[i]; }
  value_type variance(std::size_t i) const { return c_[i]; }

  template <typename OtherStorage,
            typename = detail::is_storage<OtherStorage>>
  void operator+=(const OtherStorage& other)
  {
    for (std::size_t i = 0; i < c_.size(); ++i)
      c_[i] += other.value(i);
  }

private:
  Container c_;

  template <typename Archive, typename C>
  friend void serialize(Archive&, container_storage<C>&, unsigned);
};

} // NS histogram
} // NS boost

#endif
