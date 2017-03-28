// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_CONTAINER_HPP_
#define _BOOST_HISTOGRAM_STORAGE_CONTAINER_HPP_

#include <algorithm>
#include <boost/histogram/detail/meta.hpp>
#include <cstddef>

namespace boost {
namespace histogram {

namespace detail {
template <typename Container> void init(Container &c, unsigned s) {
  if (c.size() != s) {
    c = Container(s, typename Container::value_type(0));
  } else {
    std::fill(c.begin(), c.end(), typename Container::value_type(0));
  }
}

// template signature of both std::array and boost::array
template <template <class, std::size_t> class Array, typename T, std::size_t N>
void init(Array<T, N> &c, unsigned /*unused*/) {
  std::fill(c.begin(), c.end(), typename Array<T, N>::value_type(0));
}
} // namespace detail

template <typename Container> class container_storage {
public:
  using value_type = typename Container::value_type;

  explicit container_storage(std::size_t s) { detail::init(container_, s); }

  container_storage() : container_() {} // cannot be defaulted
  container_storage(const container_storage &) = default;
  container_storage &operator=(const container_storage &) = default;
  container_storage(container_storage &&) = default;
  container_storage &operator=(container_storage &&) = default;

  template <typename OtherStorage, typename = detail::is_storage<OtherStorage>>
  explicit container_storage(const OtherStorage &other) {
    detail::init(container_, other.size());
    for (std::size_t i = 0; i < container_.size(); ++i) {
      container_[i] = other.value(i);
    }
  }

  template <typename OtherStorage>
  container_storage &operator=(const OtherStorage &other) {
    detail::init(container_, other.size());
    for (std::size_t i = 0; i < container_.size(); ++i) {
      container_[i] = other.value(i);
    }
    return *this;
  }

  std::size_t size() const { return container_.size(); }
  void increase(std::size_t i) { ++(container_[i]); }
  void increase(std::size_t i, value_type w) { container_[i] += w; }
  value_type value(std::size_t i) const { return container_[i]; }

  template <typename OtherStorage> void operator+=(const OtherStorage &other) {
    for (std::size_t i = 0; i < container_.size(); ++i) {
      container_[i] += other.value(i);
    }
  }

private:
  Container container_;

  template <typename Container1, typename Container2>
  friend bool operator==(const container_storage<Container1> &,
                         const container_storage<Container2> &);

  template <typename Archive, typename C>
  friend void serialize(Archive &, container_storage<C> &, unsigned);
};

template <typename Container1, typename Container2>
bool operator==(const container_storage<Container1> &a,
                const container_storage<Container2> &b) {
  if (a.container_.size() != b.container_.size()) {
    return false;
  }
  return std::equal(a.container_.begin(), a.container_.end(),
                    b.container_.begin());
}

} // namespace histogram
} // namespace boost

#endif
