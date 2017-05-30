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

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
}
} // namespace boost

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

  template <typename S, typename = detail::is_storage<S>>
  explicit container_storage(const S &other) {
    detail::init(container_, other.size());
    for (std::size_t i = 0; i < container_.size(); ++i) {
      container_[i] = other.value(i);
    }
  }

  template <typename S> container_storage &operator=(const S &other) {
    detail::init(container_, other.size());
    for (std::size_t i = 0; i < container_.size(); ++i) {
      container_[i] = other.value(i);
    }
    return *this;
  }

  std::size_t size() const { return container_.size(); }
  void increase(std::size_t i) { ++(container_[i]); }
  template <typename Value> void increase(std::size_t i, const Value &n) {
    container_[i] += n;
  }

  void add(std::size_t i, const value_type &val, const value_type & /* var */) {
    container_[i] += val;
  }

  value_type value(std::size_t i) const { return container_[i]; }
  value_type variance(std::size_t i) const { return container_[i]; }

  template <typename C> bool operator==(const container_storage<C> &rhs) {
    return container_.size() == rhs.container_.size() &&
           std::equal(container_.begin(), container_.end(),
                      rhs.container_.begin());
  }

private:
  Container container_;

  template <typename C> friend class container_storage;

  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

} // namespace histogram
} // namespace boost

#endif
