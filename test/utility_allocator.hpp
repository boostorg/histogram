// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_ALLOCATOR_HPP
#define BOOST_HISTOGRAM_TEST_UTILITY_ALLOCATOR_HPP

#include <algorithm>
#include <boost/core/lightweight_test.hpp>
#include <boost/core/typeinfo.hpp>
#include <unordered_map>
#include <utility>

namespace boost {
namespace histogram {

struct tracing_allocator_db : std::unordered_map<const boost::core::typeinfo*,
                                                 std::pair<std::size_t, std::size_t>> {
  template <typename T>
  std::pair<std::size_t, std::size_t>& at() {
    return this->operator[](&BOOST_CORE_TYPEID(T));
  }

  std::pair<std::size_t, std::size_t> sum;
};

template <class T>
struct tracing_allocator {
  using value_type = T;

  tracing_allocator_db* db = nullptr;

  tracing_allocator() noexcept = default;
  tracing_allocator(const tracing_allocator&) noexcept = default;
  tracing_allocator(tracing_allocator&&) noexcept = default;

  tracing_allocator(tracing_allocator_db& x) noexcept : db(&x) {}
  template <class U>
  tracing_allocator(const tracing_allocator<U>& a) noexcept : db(a.db) {}
  template <class U>
  tracing_allocator& operator=(const tracing_allocator<U>& a) noexcept {
    db = a.db;
    return *this;
  }
  ~tracing_allocator() noexcept {}

  T* allocate(std::size_t n) {
    if (db) {
      db->at<T>().first += n;
      db->sum.first += n;
    }
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  void deallocate(T* p, std::size_t n) {
    if (db) {
      db->at<T>().second += n;
      db->sum.second += n;
    }
    ::operator delete((void*)p);
  }
};

template <class T, class U>
constexpr bool operator==(const tracing_allocator<T>&,
                          const tracing_allocator<U>&) noexcept {
  return true;
}

template <class T, class U>
constexpr bool operator!=(const tracing_allocator<T>& t,
                          const tracing_allocator<U>& u) noexcept {
  return !operator==(t, u);
}

template <class T>
struct failing_allocator {
  using value_type = T;

  std::size_t nfail = 0;
  std::size_t* nbytes = nullptr;

  failing_allocator() noexcept = default;
  failing_allocator(const failing_allocator& x) noexcept = default;

  explicit failing_allocator(std::size_t nf, std::size_t& nb) noexcept
      : nfail(nf), nbytes(&nb) {}
  template <class U>
  failing_allocator(const failing_allocator<U>& a) noexcept
      : nfail(a.nfail), nbytes(a.nbytes) {}
  template <class U>
  failing_allocator& operator=(const failing_allocator<U>& a) noexcept {
    nfail = a.nfail;
    nbytes = a.nbytes;
    return *this;
  }
  ~failing_allocator() noexcept {}

  T* allocate(std::size_t n) {
    if (nbytes) {
      if (*nbytes + n * sizeof(T) > nfail) throw std::bad_alloc{};
      *nbytes += n * sizeof(T);
    }
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  void deallocate(T* p, std::size_t n) {
    if (nbytes) *nbytes -= n * sizeof(T);
    ::operator delete((void*)p);
  }
};

template <class T, class U>
constexpr bool operator==(const failing_allocator<T>&,
                          const failing_allocator<U>&) noexcept {
  return true;
}

template <class T, class U>
bool operator!=(const failing_allocator<T>& t, const failing_allocator<U>& u) noexcept {
  return !operator==(t, u);
}

} // namespace histogram
} // namespace boost

#endif
