// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP_
#define _BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP_

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/cstdint.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/histogram/weight.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
}
} // namespace boost

namespace boost {
namespace histogram {

namespace detail {

using wcount = weight_counter<double>;
using mp_int = boost::multiprecision::cpp_int;

template <typename T>
struct type_tag {};
template <>
struct type_tag<void> {
  static constexpr char value = 0;
  using next = uint8_t;
};
template <>
struct type_tag<uint8_t> {
  static constexpr char value = 1;
  using next = uint16_t;
};
template <>
struct type_tag<uint16_t> {
  static constexpr char value = 2;
  using next = uint32_t;
};
template <>
struct type_tag<uint32_t> {
  static constexpr char value = 3;
  using next = uint64_t;
};
template <>
struct type_tag<uint64_t> {
  static constexpr char value = 4;
  using next = mp_int;
};
template <>
struct type_tag<mp_int> {
  static constexpr char value = 5;
};
template <>
struct type_tag<wcount> {
  static constexpr char value = 6;
};

template <typename T>
using next_type = typename type_tag<T>::next;

template <typename T>
constexpr char type_index() {
  return type_tag<T>::value;
}

template <typename T>
bool safe_increase(T& t) {
  if (t < std::numeric_limits<T>::max()) {
    ++t;
    return true;
  }
  return false;
}

template <typename T, typename U>
bool safe_assign(T& t, const U& u) {
  if (std::numeric_limits<T>::max() < std::numeric_limits<U>::max() &&
      std::numeric_limits<T>::max() < u)
    return false;
  t = static_cast<T>(u);
  return true;
}

template <typename T, typename U>
bool safe_radd(T& t, const U& u) {
  BOOST_ASSERT(t >= 0);
  BOOST_ASSERT(u >= 0);
  // static_cast converts back from signed to unsigned integer
  if (static_cast<T>(std::numeric_limits<T>::max() - t) < u) return false;
  t += static_cast<T>(u); // static_cast to suppress conversion warning
  return true;
}

template <typename F, typename A, typename... Ts>
typename std::result_of<F(void*, A&&, Ts&&...)>::type apply(F&& f, A&& a, Ts&&... ts) {
  // this is intentionally not a switch, the if-chain is faster in benchmarks
  if (a.type == 1)
    return f(reinterpret_cast<uint8_t*>(a.ptr), std::forward<A>(a),
             std::forward<Ts>(ts)...);
  if (a.type == 2)
    return f(reinterpret_cast<uint16_t*>(a.ptr), std::forward<A>(a),
             std::forward<Ts>(ts)...);
  if (a.type == 3)
    return f(reinterpret_cast<uint32_t*>(a.ptr), std::forward<A>(a),
             std::forward<Ts>(ts)...);
  if (a.type == 4)
    return f(reinterpret_cast<uint64_t*>(a.ptr), std::forward<A>(a),
             std::forward<Ts>(ts)...);
  if (a.type == 5)
    return f(reinterpret_cast<mp_int*>(a.ptr), std::forward<A>(a),
             std::forward<Ts>(ts)...);
  if (a.type == 6)
    return f(reinterpret_cast<wcount*>(a.ptr), std::forward<A>(a),
             std::forward<Ts>(ts)...);
  // a.type == 0 is intentionally the last in the chain, because it is rarely
  // triggered
  return f(a.ptr, std::forward<A>(a), std::forward<Ts>(ts)...);
}

template <typename T, typename Buffer, typename U = T>
void create(type_tag<T>, Buffer& b, const U* init = nullptr) {
  using alloc_type = typename std::allocator_traits<
      typename Buffer::allocator_type>::template rebind_alloc<T>;
  alloc_type a(b.alloc); // rebind allocator
  using AT = std::allocator_traits<alloc_type>;
  T* p = AT::allocate(a, b.size);
  if (init) {
    for (auto it = p, end = p + b.size; it != end; ++it) AT::construct(a, it, *init++);
  } else {
    for (auto it = p, end = p + b.size; it != end; ++it) AT::construct(a, it, 0);
  }
  b.type = type_index<T>();
  b.ptr = p;
}

template <typename Buffer, typename U = void>
void create(type_tag<void>, Buffer& b, const U* init = nullptr) {
  boost::ignore_unused(init);
  BOOST_ASSERT(!init);
  b.ptr = nullptr;
  b.type = type_index<void>();
}

struct destroyer {
  template <typename T, typename Buffer>
  void operator()(T* tp, Buffer& b) {
    using alloc_type = typename std::allocator_traits<
        typename Buffer::allocator_type>::template rebind_alloc<T>;
    using AT = std::allocator_traits<alloc_type>;
    alloc_type a(b.alloc); // rebind allocator
    for (auto it = tp, end = tp + b.size; it != end; ++it) AT::destroy(a, it);
    AT::deallocate(a, tp, b.size);
  }

  template <typename Buffer>
  void operator()(void*, Buffer&) {}
};

struct replacer {
  template <typename T, typename OBuffer, typename Buffer>
  void operator()(T* optr, const OBuffer& ob, Buffer& b) {
    if (b.size == ob.size && b.type == ob.type) {
      std::copy(optr, optr + ob.size, reinterpret_cast<T*>(b.ptr));
    } else {
      apply(destroyer(), b);
      b.alloc = ob.alloc;
      b.size = ob.size;
      create(type_tag<T>(), b, optr);
    }
  }

  template <typename OBuffer, typename Buffer>
  void operator()(void*, const OBuffer& ob, Buffer& b) {
    apply(destroyer(), b);
    b.type = 0;
    b.size = ob.size;
  }
};

struct increaser {
  template <typename T, typename Buffer>
  void operator()(T* tp, Buffer& b, std::size_t i) {
    if (!safe_increase(tp[i])) {
      using U = next_type<T>;
      create(type_tag<U>(), b, tp);
      destroyer()(tp, b);
      ++reinterpret_cast<U*>(b.ptr)[i];
    }
  }

  template <typename Buffer>
  void operator()(void*, Buffer& b, std::size_t i) {
    using U = next_type<void>;
    create(type_tag<U>(), b);
    ++reinterpret_cast<U*>(b.ptr)[i];
  }

  template <typename Buffer>
  void operator()(mp_int* tp, Buffer&, std::size_t i) {
    ++tp[i];
  }

  template <typename Buffer>
  void operator()(wcount* tp, Buffer&, std::size_t i) {
    ++tp[i];
  }
};

struct adder {
  template <typename U>
  using is_convertible_to_mp_int = typename std::is_convertible<U, mp_int>::type;

  template <typename U>
  using is_integral = typename std::is_integral<U>::type;

  template <typename T, typename Buffer, typename U>
  void if_integral(std::true_type, T* tp, Buffer& b, std::size_t i, const U& x) {
    if (!safe_radd(tp[i], x)) {
      using V = next_type<T>;
      create(type_tag<V>(), b, tp);
      destroyer()(tp, b);
      operator()(reinterpret_cast<V*>(b.ptr), b, i, x);
    }
  }

  template <typename T, typename Buffer, typename U>
  void if_integral(std::false_type, T* tp, Buffer& b, std::size_t i, const U& x) {
    create(type_tag<wcount>(), b, tp);
    destroyer()(tp, b);
    operator()(reinterpret_cast<wcount*>(b.ptr), b, i, x);
  }

  template <typename T, typename Buffer, typename U>
  void operator()(T* tp, Buffer& b, std::size_t i, const U& x) {
    if_integral(is_integral<U>(), tp, b, i, x);
  }

  template <typename Buffer, typename U>
  void operator()(void*, Buffer& b, std::size_t i, const U& x) {
    using V = next_type<void>;
    create(type_tag<V>(), b);
    operator()(reinterpret_cast<V*>(b.ptr), b, i, x);
  }

  template <typename Buffer, typename U>
  void if_convertible_to_mp_int(std::true_type, mp_int* tp, Buffer&, std::size_t i,
                                const U& x) {
    tp[i] += static_cast<mp_int>(x);
  }

  template <typename Buffer, typename U>
  void if_convertible_to_mp_int(std::false_type, mp_int* tp, Buffer& b, std::size_t i,
                                const U& x) {
    create(type_tag<wcount>(), b, tp);
    destroyer()(tp, b);
    operator()(reinterpret_cast<wcount*>(b.ptr), b, i, x);
  }

  template <typename Buffer, typename U>
  void operator()(mp_int* tp, Buffer& b, std::size_t i, const U& x) {
    if_convertible_to_mp_int(is_convertible_to_mp_int<U>(), tp, b, i, x);
  }

  template <typename Buffer, typename U>
  void operator()(wcount* tp, Buffer&, std::size_t i, const U& x) {
    tp[i] += x;
  }

  template <typename Buffer>
  void operator()(wcount* tp, Buffer&, std::size_t i, const mp_int& x) {
    tp[i] += static_cast<double>(x);
  }
};

struct buffer_adder {
  template <typename T, typename OBuffer, typename Buffer>
  void operator()(T* tp, const OBuffer&, Buffer& b) {
    for (std::size_t i = 0; i < b.size; ++i) { apply(adder(), b, i, tp[i]); }
  }

  template <typename OBuffer, typename Buffer>
  void operator()(void*, const OBuffer&, Buffer&) {}
};

struct getter {
  template <typename T, typename Buffer>
  wcount operator()(T* tp, Buffer&, std::size_t i) {
    return static_cast<wcount>(tp[i]);
  }

  template <typename Buffer>
  wcount operator()(void*, Buffer&, std::size_t) {
    return static_cast<wcount>(0);
  }
};

// precondition: buffers already have same size
struct comparer {
  struct inner {
    template <typename U, typename OBuffer, typename T>
    bool operator()(const U* optr, const OBuffer& ob, const T* tp) {
      return std::equal(optr, optr + ob.size, tp);
    }

    template <typename U, typename OBuffer>
    bool operator()(const U* optr, const OBuffer& ob, const void*) {
      return std::all_of(optr, optr + ob.size, [](const U& x) { return x == 0; });
    }

    template <typename OBuffer, typename T>
    bool operator()(const void*, const OBuffer& ob, const T* tp) {
      return std::all_of(tp, tp + ob.size, [](const T& x) { return x == 0; });
    }

    template <typename OBuffer>
    bool operator()(const void*, const OBuffer&, const void*) {
      return true;
    }
  };

  template <typename T, typename Buffer, typename OBuffer>
  bool operator()(const T* tp, const Buffer& b, const OBuffer& ob) {
    BOOST_ASSERT(b.size == ob.size);
    return apply(inner(), ob, tp);
  }
};

struct multiplier {
  template <typename T, typename Buffer>
  void operator()(T* tp, Buffer& b, const double x) {
    create(type_tag<wcount>(), b, tp);
    operator()(reinterpret_cast<wcount*>(b.ptr), b, x);
  }

  template <typename Buffer>
  void operator()(void*, Buffer&, const double) {}

  template <typename Buffer>
  void operator()(wcount* tp, Buffer& b, const double x) {
    for (auto end = tp + b.size; tp != end; ++tp) *tp *= x;
  }
};

} // namespace detail

template <class Allocator>
class adaptive_storage {
public:
  using allocator_type = Allocator;
  using element_type = weight_counter<double>;
  using const_reference = element_type;

private:
  struct buffer_type {
    using allocator_type = Allocator;
    allocator_type alloc;
    char type;
    std::size_t size;
    void* ptr;
    buffer_type(std::size_t s = 0, const allocator_type& a = allocator_type())
        : alloc(a), type(0), size(s), ptr(nullptr) {}
  };

public:
  ~adaptive_storage() { detail::apply(detail::destroyer(), buffer_); }

  adaptive_storage(const adaptive_storage& o) {
    detail::apply(detail::replacer(), o.buffer_, buffer_);
  }

  adaptive_storage& operator=(const adaptive_storage& o) {
    if (this != &o) { detail::apply(detail::replacer(), o.buffer_, buffer_); }
    return *this;
  }

  adaptive_storage(adaptive_storage&& o) : buffer_(std::move(o.buffer_)) {
    o.buffer_.type = 0;
    o.buffer_.size = 0;
    o.buffer_.ptr = nullptr;
  }

  adaptive_storage& operator=(adaptive_storage&& o) {
    if (this != &o) { std::swap(buffer_, o.buffer_); }
    return *this;
  }

  template <typename S, typename = detail::requires_storage<S>>
  explicit adaptive_storage(const S& s) : buffer_(s.size(), s.get_allocator()) {
    create(detail::type_tag<detail::wcount>(), buffer_);
    auto it = reinterpret_cast<detail::wcount*>(buffer_.ptr);
    const auto end = it + size();
    std::size_t i = 0;
    while (it != end) *it++ = s[i++];
  }

  template <typename S, typename = detail::requires_storage<S>>
  adaptive_storage& operator=(const S& s) {
    // no check for self-assign needed, since S is different type
    detail::apply(detail::destroyer(), buffer_);
    buffer_.alloc = s.get_allocator();
    buffer_.size = s.size();
    create(detail::type_tag<void>(), buffer_);
    for (std::size_t i = 0; i < size(); ++i) { add(i, s[i]); }
    return *this;
  }

  explicit adaptive_storage(const allocator_type& a = allocator_type()) : buffer_(0, a) {
    detail::create(detail::type_tag<void>(), buffer_);
  }

  allocator_type get_allocator() const { return buffer_.alloc; }

  void reset(std::size_t s) {
    detail::apply(detail::destroyer(), buffer_);
    buffer_.size = s;
    create(detail::type_tag<void>(), buffer_);
  }

  std::size_t size() const { return buffer_.size; }

  void increase(std::size_t i) {
    BOOST_ASSERT(i < size());
    detail::apply(detail::increaser(), buffer_, i);
  }

  template <typename T>
  void add(std::size_t i, const T& x) {
    BOOST_ASSERT(i < size());
    detail::apply(detail::adder(), buffer_, i, x);
  }

  const_reference operator[](std::size_t i) const {
    return detail::apply(detail::getter(), buffer_, i);
  }

  bool operator==(const adaptive_storage& o) const {
    if (size() != o.size()) return false;
    return detail::apply(detail::comparer(), buffer_, o.buffer_);
  }

  // precondition: storages have same size
  adaptive_storage& operator+=(const adaptive_storage& o) {
    BOOST_ASSERT(o.size() == size());
    if (this == &o) {
      /*
        Self-adding is a special-case, because the source buffer ptr may be
        invalided by growth. We avoid this by making a copy of the source.
        This is the simplest solution, but expensive. The cost is ok, because
        self-adding is only used by the unit-tests. It does not occur
        frequently in real applications.
      */
      const auto copy = o;
      detail::apply(detail::buffer_adder(), copy.buffer_, buffer_);
    } else {
      detail::apply(detail::buffer_adder(), o.buffer_, buffer_);
    }
    return *this;
  }

  // precondition: storages have same size
  template <typename S>
  adaptive_storage& operator+=(const S& o) {
    BOOST_ASSERT(o.size() == size());
    for (std::size_t i = 0; i < size(); ++i) add(i, o[i]);
    return *this;
  }

  adaptive_storage& operator*=(const double x) {
    detail::apply(detail::multiplier(), buffer_, x);
    return *this;
  }

  // used by unit tests, not part of generic storage interface
  template <typename T>
  adaptive_storage(std::size_t s, const T* p, const allocator_type& a = allocator_type())
      : buffer_(s, a) {
    detail::create(detail::type_tag<T>(), buffer_, p);
  }

private:
  buffer_type buffer_;

  template <typename UAllocator>
  friend class adaptive_storage;
  friend class python_access;
  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

} // namespace histogram
} // namespace boost

#endif
