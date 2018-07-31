// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP_
#define _BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP_

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/cstdint.hpp>
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

// forward declaration for python
namespace boost {
namespace python {
class access;
}
} // namespace boost

namespace boost {
namespace histogram {

namespace detail {

using mp_int = ::boost::multiprecision::cpp_int;
using wcount = ::boost::histogram::weight_counter<double>;

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
  BOOST_ASSERT(u >= 0);
  if (static_cast<T>(std::numeric_limits<T>::max() - t) < u) return false;
  t += u;
  return true;
}

template <typename F, typename A, typename... Ts>
typename std::result_of<F(void*, A&&, Ts&&...)>::type apply(F&& f, A&& a,
                                                            Ts&&... ts) {
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

template <typename T, typename Buffer, typename Alloc, typename U = T>
void create(type_tag<T>, Buffer& b, Alloc& a, const U* init = nullptr) {
  using alloc_type =
      typename std::allocator_traits<Alloc>::template rebind_alloc<T>;
  alloc_type alloc(a); // rebind allocator
  T* p = std::allocator_traits<alloc_type>::allocate(alloc, b.size);
  if (init) {
    for (std::size_t i = 0; i < b.size; ++i)
      std::allocator_traits<alloc_type>::construct(alloc, p + i, init[i]);
  } else {
    for (std::size_t i = 0; i < b.size; ++i)
      std::allocator_traits<alloc_type>::construct(alloc, p + i, 0);
  }
  b.type = type_index<T>();
  b.ptr = p;
}

template <typename Buffer, typename Alloc, typename U = void>
void create(type_tag<void>, Buffer& b, Alloc&, const U* init = nullptr) {
  boost::ignore_unused(init);
  BOOST_ASSERT(!init);
  b.ptr = nullptr;
  b.type = type_index<void>();
}

struct destroyer {
  template <typename T, typename Buffer, typename Alloc>
  void operator()(T* tp, Buffer& b, Alloc& a) {
    using alloc_type =
        typename std::allocator_traits<Alloc>::template rebind_alloc<T>;
    alloc_type alloc(a); // rebind allocator
    for (std::size_t i = 0; i < b.size; ++i)
      std::allocator_traits<alloc_type>::destroy(alloc, tp + i);
    std::allocator_traits<alloc_type>::deallocate(alloc, tp, b.size);
  }

  template <typename Buffer, typename Alloc>
  void operator()(void*, Buffer&, Alloc&) {}
};

struct replacer {
  template <typename T, typename OBuffer, typename Buffer, typename Alloc>
  void operator()(T* optr, const OBuffer& ob, Buffer& b, Alloc& a) {
    if (b.size == ob.size && b.type == ob.type) {
      std::copy(optr, optr + ob.size, reinterpret_cast<T*>(b.ptr));
    } else {
      apply(destroyer(), b, a);
      b.size = ob.size;
      create(type_tag<T>(), b, a, optr);
    }
  }

  template <typename OBuffer, typename Buffer, typename Alloc>
  void operator()(void*, const OBuffer& ob, Buffer& b, Alloc& a) {
    apply(destroyer(), b, a);
    b.type = 0;
    b.size = ob.size;
  }
};

struct increaser {
  template <typename T, typename Buffer, typename Alloc>
  void operator()(T* tp, Buffer& b, Alloc& a, std::size_t i) {
    if (!safe_increase(tp[i])) {
      using U = next_type<T>;
      create(type_tag<U>(), b, a, tp);
      destroyer()(tp, b, a);
      ++reinterpret_cast<U*>(b.ptr)[i];
    }
  }

  template <typename Buffer, typename Alloc>
  void operator()(void*, Buffer& b, Alloc& a, std::size_t i) {
    using U = next_type<void>;
    create(type_tag<U>(), b, a);
    ++reinterpret_cast<U*>(b.ptr)[i];
  }

  template <typename Buffer, typename Alloc>
  void operator()(mp_int* tp, Buffer&, Alloc&, std::size_t i) {
    ++tp[i];
  }

  template <typename Buffer, typename Alloc>
  void operator()(wcount* tp, Buffer&, Alloc&, std::size_t i) {
    ++tp[i];
  }
};

struct adder {
  template <typename U>
  using is_convertible_to_mp_int =
      typename std::is_convertible<U, mp_int>::type;

  template <typename U>
  using is_integral = typename std::is_integral<U>::type;

  template <typename T, typename Buffer, typename Alloc, typename U>
  void if_integral(std::true_type, T* tp, Buffer& b, Alloc& a, std::size_t i,
                   const U& x) {
    if (!safe_radd(tp[i], x)) {
      using V = next_type<T>;
      create(type_tag<V>(), b, a, tp);
      destroyer()(tp, b, a);
      operator()(reinterpret_cast<V*>(b.ptr), b, a, i, x);
    }
  }

  template <typename T, typename Buffer, typename Alloc, typename U>
  void if_integral(std::false_type, T* tp, Buffer& b, Alloc& a, std::size_t i,
                   const U& x) {
    create(type_tag<wcount>(), b, a, tp);
    destroyer()(tp, b, a);
    operator()(reinterpret_cast<wcount*>(b.ptr), b, a, i, x);
  }

  template <typename T, typename Buffer, typename Alloc, typename U>
  void operator()(T* tp, Buffer& b, Alloc& a, std::size_t i, const U& x) {
    if_integral(is_integral<U>(), tp, b, a, i, x);
  }

  template <typename Buffer, typename Alloc, typename U>
  void operator()(void*, Buffer& b, Alloc& a, std::size_t i, const U& x) {
    using V = next_type<void>;
    create(type_tag<V>(), b, a);
    operator()(reinterpret_cast<V*>(b.ptr), b, a, i, x);
  }

  template <typename Buffer, typename Alloc, typename U>
  void if_convertible_to_mp_int(std::true_type, mp_int* tp, Buffer&, Alloc&,
                                std::size_t i, const U& x) {
    tp[i] += static_cast<mp_int>(x);
  }

  template <typename Buffer, typename Alloc, typename U>
  void if_convertible_to_mp_int(std::false_type, mp_int* tp, Buffer& b,
                                Alloc& a, std::size_t i, const U& x) {
    create(type_tag<wcount>(), b, a, tp);
    destroyer()(tp, b, a);
    operator()(reinterpret_cast<wcount*>(b.ptr), b, a, i, x);
  }

  template <typename Buffer, typename Alloc, typename U>
  void operator()(mp_int* tp, Buffer& b, Alloc& a, std::size_t i,
                  const U& x) {
    if_convertible_to_mp_int(is_convertible_to_mp_int<U>(), tp, b, a, i, x);
  }

  template <typename Buffer, typename Alloc, typename U>
  void operator()(wcount* tp, Buffer&, Alloc&, std::size_t i, const U& x) {
    tp[i] += x;
  }

  template <typename Buffer, typename Alloc>
  void operator()(wcount* tp, Buffer&, Alloc&, std::size_t i,
                  const mp_int& x) {
    tp[i] += static_cast<double>(x);
  }
};

struct buffer_adder {
  template <typename T, typename OBuffer, typename Buffer, typename Alloc>
  void operator()(T* tp, const OBuffer&, Buffer& b, Alloc& a) {
    for (std::size_t i = 0; i < b.size; ++i) {
      apply(adder(), b, a, i, tp[i]);
    }
  }

  template <typename OBuffer, typename Buffer, typename Alloc>
  void operator()(void*, const OBuffer&, Buffer&, Alloc&) {}
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
      return std::all_of(optr, optr + ob.size,
                         [](const U& x) { return x == 0; });
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
  template <typename T, typename Buffer, typename Alloc>
  void operator()(T* tp, Buffer& b, Alloc& a, const double x) {
    create(type_tag<wcount>(), b, a, tp);
    operator()(reinterpret_cast<wcount*>(b.ptr), b, a, x);
  }

  template <typename Buffer, typename Alloc>
  void operator()(void*, Buffer&, Alloc&, const double) {}

  template <typename Buffer, typename Alloc>
  void operator()(wcount* tp, Buffer& b, Alloc&, const double x) {
    for (std::size_t i = 0; i < b.size; ++i) tp[i] *= x;
  }
};

} // namespace detail

template <class Alloc>
class adaptive_storage {
  struct buffer_type {
    char type;
    std::size_t size;
    void* ptr;
    buffer_type(std::size_t s = 0) : type(0), size(s), ptr(nullptr) {}
  };

public:
  using element_type = detail::wcount;
  using const_reference = element_type;

  adaptive_storage() = default;

  explicit adaptive_storage(const Alloc& a) : alloc_(a) {}

  adaptive_storage(const adaptive_storage& o) : alloc_(o.alloc_) {
    detail::apply(detail::replacer(), o.buffer_, buffer_, alloc_);
  }

  adaptive_storage& operator=(const adaptive_storage& o) {
    if (this != &o) {
      alloc_ = o.alloc_;
      detail::apply(detail::replacer(), o.buffer_, buffer_, alloc_);
    }
    return *this;
  }

  adaptive_storage(adaptive_storage&& o)
      : alloc_(std::move(o.alloc_)), buffer_(std::move(o.buffer_)) {
    o.buffer_.type = 0;
    o.buffer_.size = 0;
    o.buffer_.ptr = nullptr;
  }

  adaptive_storage& operator=(adaptive_storage&& o) {
    if (this != &o) {
      std::swap(alloc_, o.alloc_);
      std::swap(buffer_, o.buffer_);
    }
    return *this;
  }

  ~adaptive_storage() { detail::apply(detail::destroyer(), buffer_, alloc_); }

  template <typename S>
  explicit adaptive_storage(const S& s) : buffer_(s.size()) {
    create(detail::type_tag<detail::wcount>(), buffer_, alloc_);
    for (std::size_t i = 0; i < size(); ++i) {
      reinterpret_cast<detail::wcount*>(buffer_.ptr)[i] = s[i];
    }
  }

  template <typename S>
  explicit adaptive_storage(const S& s, const Alloc& a)
      : alloc_(a), buffer_(s.size()) {
    create(detail::type_tag<detail::wcount>(), buffer_, alloc_);
    for (std::size_t i = 0; i < size(); ++i) {
      reinterpret_cast<detail::wcount*>(buffer_.ptr)[i] = s[i];
    }
  }

  template <typename S>
  adaptive_storage& operator=(const S& s) {
    // no check for self-assign needed, since S is different type
    detail::apply(detail::destroyer(), buffer_, alloc_);
    buffer_.size = s.size();
    create(detail::type_tag<void>(), buffer_, alloc_);
    for (std::size_t i = 0; i < size(); ++i) { add(i, s[i]); }
    return *this;
  }

  explicit adaptive_storage(std::size_t s) : buffer_(s) {
    detail::create(detail::type_tag<void>(), buffer_, alloc_);
  }

  explicit adaptive_storage(std::size_t s, const Alloc& a)
      : alloc_(a), buffer_(s) {
    detail::create(detail::type_tag<void>(), buffer_, alloc_);
  }

  // used by unit tests, not part of generic storage interface
  template <typename T>
  adaptive_storage(std::size_t s, const T* p) : buffer_(s) {
    detail::create(detail::type_tag<T>(), buffer_, alloc_, p);
  }

  std::size_t size() const { return buffer_.size; }

  void increase(std::size_t i) {
    BOOST_ASSERT(i < size());
    detail::apply(detail::increaser(), buffer_, alloc_, i);
  }

  template <typename T>
  void add(std::size_t i, const T& x) {
    BOOST_ASSERT(i < size());
    detail::apply(detail::adder(), buffer_, alloc_, i, x);
  }

  template <typename T>
  void add(std::size_t i, const detail::weight<T>& x) {
    BOOST_ASSERT(i < size());
    detail::apply(detail::adder(), buffer_, alloc_, i, x);
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
        Self-adding needs to be special-cased, because the source buffer ptr
        may be invalided by growth. We avoid this by making a copy of the
        source. This is a simple, but expensive solution. This is ok, because
        self-adding is mostly used in the unit-tests to grow a histogram
        quickly. It does not occur frequently in real applications.
      */
      const auto o_copy = o;
      detail::apply(detail::buffer_adder(), o_copy.buffer_, buffer_, alloc_);
    } else {
      detail::apply(detail::buffer_adder(), o.buffer_, buffer_, alloc_);
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
    detail::apply(detail::multiplier(), buffer_, alloc_, x);
    return *this;
  }

private:
  Alloc alloc_;
  buffer_type buffer_;

  friend class ::boost::python::access;
  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

} // namespace histogram
} // namespace boost

#endif
