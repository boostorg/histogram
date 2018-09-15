// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP
#define BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/cstdint.hpp>
#include <boost/histogram/detail/buffer.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/histogram/weight.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/mp11.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>

namespace boost {
namespace histogram {

namespace detail {
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
}

template <class Allocator>
struct adaptive_storage {
  static_assert(
      std::is_same<typename std::allocator_traits<Allocator>::pointer,
                   typename std::allocator_traits<Allocator>::value_type*>::value,
      "adaptive_storage requires allocator with trivial pointer type");

  using allocator_type = Allocator;
  using element_type = weight_counter<double>;
  using scale_type = double;
  using const_reference = element_type;

  using wcount = weight_counter<double>;
  using mp_int = boost::multiprecision::number<
    boost::multiprecision::cpp_int_backend<
      0, 0, boost::multiprecision::signed_magnitude,
      boost::multiprecision::unchecked,
      typename std::allocator_traits<Allocator>::template rebind_alloc<
        boost::multiprecision::limb_type
      >
    >
  >;

  using types = mp11::mp_list<void, uint8_t, uint16_t, uint32_t,uint64_t,
                              mp_int, wcount>;

  template <typename T>
  static constexpr char type_index() {
    return static_cast<char>(mp11::mp_find<types, T>::value);
  }

  struct buffer_type {
    allocator_type alloc;
    char type;
    std::size_t size;
    void* ptr;
    buffer_type(std::size_t s = 0, const allocator_type& a = allocator_type())
        : alloc(a), type(0), size(s), ptr(nullptr) {}

    template <typename T, typename U>
    T* create_impl(T*, const U* init) {
      using alloc_type = typename std::allocator_traits<
        allocator_type>::template rebind_alloc<T>;
      alloc_type a(alloc); // rebind allocator
      return init ? detail::create_buffer_from_iter(a, size, init) :
                    detail::create_buffer(a, size, 0);
    }

    template <typename U = mp_int>
    mp_int* create_impl(mp_int*, const U* init) {
      using alloc_type = typename std::allocator_traits<
        allocator_type>::template rebind_alloc<mp_int>;
      alloc_type a(alloc); // rebound allocator for buffer
      // mp_int has no ctor with an allocator instance, cannot pass state :(
      // typename mp_int::backend_type::allocator_type a2(alloc);
      return init ? detail::create_buffer_from_iter(a, size, init) :
                    detail::create_buffer(a, size, 0);
    }

    void* create_impl(void*, const void* init) {
      boost::ignore_unused(init);
      BOOST_ASSERT(!init); // init is always a nullptr in this specialization
      return nullptr;
    }

    template <typename T, typename U = T>
    T* create(const U* init = nullptr) {
      return create_impl(static_cast<T*>(nullptr), init);
    }

    template <typename T>
    void set(T* p) {
      type = type_index<T>();
      ptr = p;
    }
  };

  ~adaptive_storage() { apply(destroyer(), buffer); }

  adaptive_storage(const adaptive_storage& o) {
    apply(replacer(), o.buffer, buffer);
  }

  adaptive_storage& operator=(const adaptive_storage& o) {
    if (this != &o) { apply(replacer(), o.buffer, buffer); }
    return *this;
  }

  adaptive_storage(adaptive_storage&& o) : buffer(std::move(o.buffer)) {
    o.buffer.type = 0;
    o.buffer.size = 0;
    o.buffer.ptr = nullptr;
  }

  adaptive_storage& operator=(adaptive_storage&& o) {
    if (this != &o) { std::swap(buffer, o.buffer); }
    return *this;
  }

  template <typename S, typename = detail::requires_storage<S>>
  explicit adaptive_storage(const S& s) : buffer(s.size(), s.get_allocator()) {
    buffer.set(buffer.template create<wcount>());
    auto it = static_cast<wcount*>(buffer.ptr);
    const auto end = it + size();
    std::size_t i = 0;
    while (it != end) *it++ = s[i++];
  }

  template <typename S, typename = detail::requires_storage<S>>
  adaptive_storage& operator=(const S& s) {
    // no check for self-assign needed, since S is different type
    apply(destroyer(), buffer);
    buffer.alloc = s.get_allocator();
    buffer.size = s.size();
    buffer.set(buffer.template create<void>());
    for (std::size_t i = 0; i < size(); ++i) { add(i, s[i]); }
    return *this;
  }

  explicit adaptive_storage(const allocator_type& a = allocator_type()) : buffer(0, a) {
    buffer.set(buffer.template create<void>());
  }

  allocator_type get_allocator() const { return buffer.alloc; }

  void reset(std::size_t s) {
    apply(destroyer(), buffer);
    buffer.size = s;
    buffer.set(buffer.template create<void>());
  }

  std::size_t size() const { return buffer.size; }

  void increase(std::size_t i) {
    BOOST_ASSERT(i < size());
    apply(increaser(), buffer, i);
  }

  template <typename T>
  void add(std::size_t i, const T& x) {
    BOOST_ASSERT(i < size());
    apply(adder(), buffer, i, x);
  }

  const_reference operator[](std::size_t i) const {
    return apply(getter(), buffer, i);
  }

  bool operator==(const adaptive_storage& o) const {
    if (size() != o.size()) return false;
    return apply(comparer(), buffer, o.buffer);
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
      apply(buffer_adder(), copy.buffer, buffer);
    } else {
      apply(buffer_adder(), o.buffer, buffer);
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
    apply(multiplier(), buffer, x);
    return *this;
  }

  // used by unit tests, not part of generic storage interface
  template <typename T>
  adaptive_storage(std::size_t s, const T* p, const allocator_type& a = allocator_type())
      : buffer(s, a) {
    buffer.set(buffer.template create<T>(p));
  }

  struct destroyer {
    template <typename T, typename Buffer>
    void operator()(T* tp, Buffer& b) {
      using alloc_type = typename std::allocator_traits<
        allocator_type>::template rebind_alloc<T>;
      alloc_type a(b.alloc); // rebind allocator
      detail::destroy_buffer(a, tp, b.size);
    }

    template <typename Buffer>
    void operator()(void*, Buffer&) {}
  };

  template <typename F, typename B, typename... Ts>
  static typename std::result_of<F(void*, B&&, Ts&&...)>::type apply(F&& f, B&& b, Ts&&... ts) {
    // this is intentionally not a switch, the if-chain is faster in benchmarks
    if (b.type == type_index<uint8_t>())
      return f(reinterpret_cast<uint8_t*>(b.ptr), std::forward<B>(b),
               std::forward<Ts>(ts)...);
    if (b.type == type_index<uint16_t>())
      return f(reinterpret_cast<uint16_t*>(b.ptr), std::forward<B>(b),
               std::forward<Ts>(ts)...);
    if (b.type == type_index<uint32_t>())
      return f(reinterpret_cast<uint32_t*>(b.ptr), std::forward<B>(b),
               std::forward<Ts>(ts)...);
    if (b.type == type_index<uint64_t>())
      return f(reinterpret_cast<uint64_t*>(b.ptr), std::forward<B>(b),
               std::forward<Ts>(ts)...);
    if (b.type == type_index<mp_int>())
      return f(reinterpret_cast<mp_int*>(b.ptr), std::forward<B>(b),
               std::forward<Ts>(ts)...);
    if (b.type == type_index<wcount>())
      return f(reinterpret_cast<wcount*>(b.ptr), std::forward<B>(b),
               std::forward<Ts>(ts)...);
    // b.type == 0 is intentionally the last in the chain, because it is rarely
    // triggered
    return f(b.ptr, std::forward<B>(b), std::forward<Ts>(ts)...);
  }

  struct replacer {
    template <typename T, typename OBuffer, typename Buffer>
    void operator()(T* optr, const OBuffer& ob, Buffer& b) {
      if (b.size == ob.size && b.type == ob.type) {
        std::copy(optr, optr + ob.size, reinterpret_cast<T*>(b.ptr));
      } else {
        apply(destroyer(), b);
        b.alloc = ob.alloc;
        b.size = ob.size;
        b.set(b.template create<T>(optr));
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
      if (!detail::safe_increase(tp[i])) {
        using U = mp11::mp_at_c<types, (type_index<T>() + 1)>;
        U* ptr = b.template create<U>(tp);
        destroyer()(tp, b);
        b.set(ptr);
        ++reinterpret_cast<U*>(b.ptr)[i];
      }
    }

    template <typename Buffer>
    void operator()(void*, Buffer& b, std::size_t i) {
      using U = mp11::mp_at_c<types, 1>;
      b.set(b.template create<U>());
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
    template <typename Buffer, typename U>
    void if_U_is_integral(std::true_type, mp_int* tp, Buffer&, std::size_t i, const U& x) {
      tp[i] += static_cast<mp_int>(x);
    }

    template <typename T, typename Buffer, typename U>
    void if_U_is_integral(std::true_type, T* tp, Buffer& b, std::size_t i, const U& x) {
      if (!detail::safe_radd(tp[i], x)) {
        using V = mp11::mp_at_c<types, (type_index<T>() + 1)>;
        auto ptr = b.template create<V>(tp);
        destroyer()(tp, b);
        b.set(ptr);
        if_U_is_integral(std::true_type(), static_cast<V*>(b.ptr), b, i, x);
      }
    }

    template <typename T, typename Buffer, typename U>
    void if_U_is_integral(std::false_type, T* tp, Buffer& b, std::size_t i, const U& x) {
      auto ptr = b.template create<wcount>(tp);
      destroyer()(tp, b);
      b.set(ptr);
      operator()(static_cast<wcount*>(b.ptr), b, i, x);
    }

    template <typename T, typename Buffer, typename U>
    void operator()(T* tp, Buffer& b, std::size_t i, const U& x) {
      if_U_is_integral(std::integral_constant<bool,
                       (std::is_integral<U>::value ||
                        std::is_same<U, mp_int>::value)>(), tp, b, i, x);
    }

    template <typename Buffer, typename U>
    void operator()(void*, Buffer& b, std::size_t i, const U& x) {
      using V = mp11::mp_at_c<types, 1>;
      b.set(b.template create<V>());
      operator()(reinterpret_cast<V*>(b.ptr), b, i, x);
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
      auto ptr = b.template create<wcount>(tp);
      destroyer()(tp, b);
      b.set(ptr);
      operator()(reinterpret_cast<wcount*>(b.ptr), b, x);
    }

    template <typename Buffer>
    void operator()(void*, Buffer&, const double) {}

    template <typename Buffer>
    void operator()(wcount* tp, Buffer& b, const double x) {
      for (auto end = tp + b.size; tp != end; ++tp) *tp *= x;
    }
  };

  buffer_type buffer;
};

} // namespace histogram
} // namespace boost

#endif
