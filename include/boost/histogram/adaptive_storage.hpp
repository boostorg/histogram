// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ADAPTIVE_STORAGE_HPP
#define BOOST_HISTOGRAM_ADAPTIVE_STORAGE_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/cstdint.hpp>
#include <boost/histogram/detail/buffer.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#if defined BOOST_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
// warning-ignore required in Boost-1.66 for cpp_int.hpp:822
#include <boost/multiprecision/cpp_int.hpp>
#if defined BOOST_CLANG
#pragma clang diagnostic pop
#endif
#include <limits>
#include <memory>
#include <type_traits>

namespace boost {
namespace histogram {

namespace detail {

template <class T>
struct is_unsigned_integral : mp11::mp_and<std::is_integral<T>, std::is_unsigned<T>> {};

template <class T>
bool safe_increment(T& t) {
  if (t < std::numeric_limits<T>::max()) {
    ++t;
    return true;
  }
  return false;
}

template <class T, class U>
bool safe_radd(std::true_type, T& t, const U& u) {
  if ((std::numeric_limits<T>::max() - t) >= u) {
    t += static_cast<T>(u); // static_cast to suppress conversion warning
    return true;
  }
  return false;
}

template <class T, class U>
bool safe_radd(std::false_type, T& t, const U& u) {
  static_assert(std::is_integral<U>::value, "U must be integral type");
  if (u >= 0) {
    if (std::numeric_limits<T>::max() - t >= static_cast<std::make_unsigned_t<U>>(u)) {
      t += static_cast<T>(u);
      return true;
    }
    return false;
  }
  if (t >= static_cast<std::make_unsigned_t<U>>(-u)) {
    t -= static_cast<T>(-u);
    return true;
  }
  return false;
}

template <class T, class U>
bool safe_radd(std::false_type, T& t, const boost::multiprecision::number<U>& u) {
  if (u >= 0) {
    if (std::numeric_limits<T>::max() - t > u) {
      t += static_cast<T>(u);
      return true;
    }
  }
  if (t + u >= 0) {
    t -= static_cast<T>(-u);
    return true;
  }
  return false;
}

template <class T, class U>
bool safe_radd(T& t, const U& u) {
  static_assert(is_unsigned_integral<T>::value, "T must be unsigned integral type");
  return safe_radd(is_unsigned_integral<U>{}, t, u);
}

} // namespace detail

/**
  Memory-efficient storage for integral counters which cannot overflow.

  This storage provides a no-overflow-guarantee if it is filled with integral weights
  only. This storage implementation keeps a contiguous array of elemental counters, one
  for each cell. If an operation is requested, which would overflow a counter, the whole
  array is replaced with another of a wider integral type, then the operation is executed.
  The storage uses integers of 8, 16, 32, 64 bits, and then switches to a multiprecision
  integral type, cpp_int from
  [Boost.Multiprecision](https://www.boost.org/doc/libs/develop/libs/multiprecision/doc/html/index.html).

  A scaling operation or adding a floating point number turns the elements into doubles,
  which voids the no-overflow-guarantee.
*/
template <class Allocator>
class adaptive_storage {
  static_assert(
      std::is_same<typename std::allocator_traits<Allocator>::pointer,
                   typename std::allocator_traits<Allocator>::value_type*>::value,
      "adaptive_storage requires allocator with trivial pointer type");

public:
  struct storage_tag {};
  using allocator_type = Allocator;
  using value_type = double;

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED
  using mp_int = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<
      0, 0, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked,
      typename std::allocator_traits<Allocator>::template rebind_alloc<
          boost::multiprecision::limb_type>>>;

  static inline auto negate(const uint8_t& t) { return -static_cast<int16_t>(t); }
  static inline auto negate(const uint16_t& t) { return -static_cast<int32_t>(t); }
  static inline auto negate(const uint32_t& t) { return -static_cast<int64_t>(t); }
  static inline mp_int negate(const uint64_t& t) { return -mp_int(t); }
  static inline mp_int negate(const mp_int& t) { return -t; }
  template <class T>
  static inline auto negate(const T& t) {
    return -t;
  }
#endif

private:
  struct equal_to {
    bool operator()(const mp_int& a, const mp_int& b) const noexcept { return a == b; }

    template <class T>
    bool operator()(const mp_int& a, const T& b) const noexcept {
      return static_cast<double>(a) == b;
    }

    template <class T>
    bool operator()(const T& a, const mp_int& b) const noexcept {
      return operator()(b, a);
    }

    template <class T>
    bool operator()(const T& a, const T& b) const noexcept {
      return a == b;
    }

    template <class T, class U>
    bool operator()(const T& a, const U& b) const noexcept {
      return static_cast<double>(a) == static_cast<double>(b);
    }
  };

  struct less {
    bool operator()(const mp_int& a, const mp_int& b) const noexcept { return a < b; }

    template <class T>
    bool operator()(const mp_int& a, const T& b) const noexcept {
      return static_cast<double>(a) < b;
    }

    template <class T>
    bool operator()(const T& a, const mp_int& b) const noexcept {
      return a < static_cast<double>(b);
    }

    template <class T>
    bool operator()(const T& a, const T& b) const noexcept {
      return a < b;
    }

    template <class T, class U>
    bool operator()(const T& a, const U& b) const noexcept {
      return static_cast<double>(a) < static_cast<double>(b);
    }
  };

  struct greater {
    bool operator()(const mp_int& a, const mp_int& b) const noexcept { return a > b; }

    template <class T>
    bool operator()(const mp_int& a, const T& b) const noexcept {
      return static_cast<double>(a) > b;
    }

    template <class T>
    bool operator()(const T& a, const mp_int& b) const noexcept {
      return a > static_cast<double>(b);
    }

    template <class T>
    bool operator()(const T& a, const T& b) const noexcept {
      return a > b;
    }

    template <class T, class U>
    bool operator()(const T& a, const U& b) const noexcept {
      return static_cast<double>(a) > static_cast<double>(b);
    }
  };

  using types = mp11::mp_list<uint8_t, uint16_t, uint32_t, uint64_t, mp_int, double>;

  template <class T>
  static constexpr char type_index() {
    return static_cast<char>(mp11::mp_find<types, T>::value);
  }

  struct buffer_type {
    allocator_type alloc;
    std::size_t size;
    char type = 0;
    void* ptr = nullptr;

    // no allocation here
    buffer_type(std::size_t s = 0, const allocator_type& a = allocator_type()) noexcept
        : alloc(a), size(s) {}

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable : 4244) // possible loss of data
#endif

    template <class T, class U>
    T* create_impl(T*, const U* init) {
      using alloc_type =
          typename std::allocator_traits<allocator_type>::template rebind_alloc<T>;
      alloc_type a(alloc); // rebind allocator
      return init ? detail::create_buffer_from_iter(a, size, init)
                  : detail::create_buffer(a, size, 0);
    }

    // create_impl: no specialization for mp_int, it has no ctor which accepts
    // allocator, cannot pass state :(

    template <class T, class U = T>
    T* create(const U* init = nullptr) {
      return create_impl(static_cast<T*>(nullptr), init);
    }

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

    template <class T>
    void set(T* p) {
      type = type_index<T>();
      ptr = p;
    }
  };

  template <class F, class B, class... Ts>
  static decltype(auto) apply(F&& f, B&& b, Ts&&... ts) {
    // this is intentionally not a switch, the if-chain is faster in benchmarks
    if (b.type == type_index<uint8_t>())
      return f(reinterpret_cast<uint8_t*>(b.ptr), b, std::forward<Ts>(ts)...);
    if (b.type == type_index<uint16_t>())
      return f(reinterpret_cast<uint16_t*>(b.ptr), b, std::forward<Ts>(ts)...);
    if (b.type == type_index<uint32_t>())
      return f(reinterpret_cast<uint32_t*>(b.ptr), b, std::forward<Ts>(ts)...);
    if (b.type == type_index<uint64_t>())
      return f(reinterpret_cast<uint64_t*>(b.ptr), b, std::forward<Ts>(ts)...);
    if (b.type == type_index<mp_int>())
      return f(reinterpret_cast<mp_int*>(b.ptr), b, std::forward<Ts>(ts)...);
    return f(reinterpret_cast<double*>(b.ptr), b, std::forward<Ts>(ts)...);
  }

  template <class Buffer>
  class reference_t {
  public:
    reference_t(Buffer* b, std::size_t i) : buffer_(b), idx_(i) {}

    reference_t(const reference_t&) = default;
    reference_t& operator=(const reference_t&) = delete; // references do not rebind
    reference_t& operator=(reference_t&&) = delete;      // references do not rebind

    template <class U>
    bool operator==(const U& rhs) const {
      return op<equal_to>(rhs);
    }

    template <class U>
    bool operator<(const U& rhs) const {
      return op<less>(rhs);
    }

    template <class U>
    bool operator>(const U& rhs) const {
      return op<greater>(rhs);
    }

    template <class U>
    bool operator!=(const U& rhs) const {
      return !operator==(rhs);
    }

    template <class U>
    bool operator>=(const U& rhs) const {
      return !operator<(rhs);
    }

    template <class U>
    bool operator<=(const U& rhs) const {
      return !operator<(rhs);
    }

    operator double() const { return apply(getter(), *buffer_, idx_); }

  protected:
    template <class Binary, class U>
    bool op(const reference_t<U>& rhs) const {
      const auto i = idx_;
      const auto j = rhs.idx_;
      return apply(
          [i, j, &rhs](const auto* ptr, const Buffer&) {
            const auto& pi = ptr[i];
            return apply([&pi, j](const auto* q, const U&) { return Binary()(pi, q[j]); },
                         *rhs.buffer_);
          },
          *buffer_);
    }

    template <class Binary, class U>
    bool op(const U& rhs) const {
      const auto i = idx_;
      return apply(
          [i, &rhs](const auto* tp, const auto&) { return Binary()(tp[i], rhs); },
          *buffer_);
    }

    template <class U>
    friend class reference_t;

    Buffer* buffer_;
    std::size_t idx_;
  };

public:
  using const_reference = reference_t<const buffer_type>;

  class reference : public reference_t<buffer_type> {
    using base_type = reference_t<buffer_type>;

  public:
    using base_type::base_type;

    reference& operator=(const reference& t) {
      apply(setter(), *base_type::buffer_, base_type::idx_, t);
      return *this;
    }

    template <class T>
    reference& operator=(const T& t) {
      apply(setter(), *base_type::buffer_, base_type::idx_, t);
      return *this;
    }

    template <class T>
    reference& operator+=(const T& t) {
      apply(adder(), *base_type::buffer_, base_type::idx_, t);
      return *this;
    }

    template <class T>
    reference& operator-=(const T& t) {
      apply(adder(), *base_type::buffer_, base_type::idx_, negate(t));
      return *this;
    }

    reference& operator++() {
      apply(incrementor(), *base_type::buffer_, base_type::idx_);
      return *this;
    }
  };

private:
  template <class Value, class Reference, class Buffer>
  class iterator_t
      : public boost::iterator_adaptor<iterator_t<Value, Reference, Buffer>, std::size_t,
                                       Value, std::random_access_iterator_tag, Reference,
                                       std::ptrdiff_t> {

  public:
    iterator_t() = default;
    template <class V, class R, class B>
    iterator_t(const iterator_t<V, R, B>& it)
        : iterator_t::iterator_adaptor_(it.base()), buffer_(it.buffer_) {}
    iterator_t(Buffer* b, std::size_t i) noexcept
        : iterator_t::iterator_adaptor_(i), buffer_(b) {}

  protected:
    template <class V, class R, class B>
    bool equal(const iterator_t<V, R, B>& rhs) const noexcept {
      return buffer_ == rhs.buffer_ && this->base() == rhs.base();
    }
    Reference dereference() const { return {buffer_, this->base()}; }

    friend class ::boost::iterator_core_access;
    template <class V, class R, class B>
    friend class iterator_t;

  private:
    Buffer* buffer_ = nullptr;
  };

public:
  using const_iterator = iterator_t<const value_type, const_reference, const buffer_type>;
  using iterator = iterator_t<value_type, reference, buffer_type>;

  ~adaptive_storage() { apply(destroyer(), buffer); }

  adaptive_storage(const adaptive_storage& o) { apply(setter(), o.buffer, buffer); }

  adaptive_storage& operator=(const adaptive_storage& o) {
    if (this != &o) { apply(setter(), o.buffer, buffer); }
    return *this;
  }

  adaptive_storage(adaptive_storage&& o) : buffer(std::move(o.buffer)) {
    o.buffer.size = 0;
    o.buffer.ptr = nullptr;
  }

  adaptive_storage& operator=(adaptive_storage&& o) {
    if (this != &o) { std::swap(buffer, o.buffer); }
    return *this;
  }

  template <class T>
  adaptive_storage(const storage_adaptor<T>& s) : buffer(s.size()) {
    buffer.set(buffer.template create<uint8_t>());
    std::size_t i = 0;
    for (auto&& x : s) apply(setter(), buffer, i++, x);
  }

  template <class C, class = detail::requires_iterable<C>>
  adaptive_storage& operator=(const C& s) {
    apply(destroyer(), buffer);
    buffer.size = s.size();
    using V = detail::naked<decltype(s[0])>;
    const unsigned ti = type_index<V>();
    if (ti < mp11::mp_size<types>::value)
      buffer.set(buffer.template create<V>());
    else
      buffer.set(buffer.template create<double>());
    std::size_t i = 0;
    for (auto&& x : s) apply(setter(), buffer, i++, x);
    return *this;
  }

  explicit adaptive_storage(const allocator_type& a = allocator_type()) : buffer(0, a) {}

  allocator_type get_allocator() const { return buffer.alloc; }

  void reset(std::size_t s) {
    apply(destroyer(), buffer);
    buffer.size = s;
    try {
      buffer.set(buffer.template create<uint8_t>());
    } catch (...) {
      buffer.size = 0;
      buffer.type = 0;
      buffer.ptr = nullptr;
      throw;
    }
  }

  std::size_t size() const noexcept { return buffer.size; }

  reference operator[](std::size_t i) noexcept { return {&buffer, i}; }
  const_reference operator[](std::size_t i) const noexcept { return {&buffer, i}; }

  bool operator==(const adaptive_storage& o) const noexcept {
    if (size() != o.size()) return false;
    return apply(
        [&o](const auto* ptr, const buffer_type&) {
          return apply(
              [ptr](const auto* optr, const buffer_type& ob) {
                return std::equal(ptr, ptr + ob.size, optr, equal_to());
              },
              o.buffer);
        },
        buffer);
  }

  template <class T>
  bool operator==(const T& o) const {
    if (size() != o.size()) return false;
    return apply(
        [&o](const auto* ptr, const buffer_type&) {
          return std::equal(ptr, ptr + o.size(), std::begin(o), equal_to());
        },
        buffer);
  }

  adaptive_storage& operator+=(const adaptive_storage& rhs) {
    BOOST_ASSERT(size() == rhs.size());
    if (this == &rhs) {
      /*
        Self-adding is a special-case, because the source buffer ptr may be invalided by
        growth. We avoid this by making a copy of the source. This is the simplest
        solution, but expensive. The cost is ok, since this is not likely to happen a lot
        in user code, but the unit-tests do this a lot.
      */
      const auto copy = rhs;
      apply(buffer_adder(), copy.buffer, buffer);
    } else {
      apply(buffer_adder(), rhs.buffer, buffer);
    }
    return *this;
  }

  template <class S>
  adaptive_storage& operator+=(const S& rhs) {
    BOOST_ASSERT(size() == rhs.size());
    for (std::size_t i = 0, n = size(); i < n; ++i) (*this)[i] += rhs[i];
    return *this;
  }

  adaptive_storage& operator-=(const adaptive_storage& rhs) {
    BOOST_ASSERT(size() == rhs.size());
    if (this == &rhs) {
      /*
        Self-subtracting is a special-case, because the source buffer ptr may be
        invalided by growth. We avoid this by making a copy of the source.
      */
      const auto copy = rhs;
      apply(buffer_subtractor(), copy.buffer, buffer);
    } else {
      apply(buffer_subtractor(), rhs.buffer, buffer);
    }
    return *this;
  }

  template <class S>
  adaptive_storage& operator-=(const S& rhs) {
    BOOST_ASSERT(size() == rhs.size());
    for (std::size_t i = 0, n = size(); i < n; ++i) (*this)[i] -= rhs[i];
    return *this;
  }

  adaptive_storage& operator*=(const double x) {
    apply(multiplier(), buffer, x);
    return *this;
  }

  iterator begin() noexcept { return {&buffer, 0}; }
  iterator end() noexcept { return {&buffer, size()}; }
  const_iterator begin() const noexcept { return {&buffer, 0}; }
  const_iterator end() const noexcept { return {&buffer, size()}; }

  /// @private used by unit tests, not part of generic storage interface
  template <class T>
  adaptive_storage(std::size_t s, const T* p, const allocator_type& a = allocator_type())
      : buffer(s, a) {
    buffer.set(buffer.template create<T>(p));
  }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  struct destroyer {
    template <class T, class Buffer>
    void operator()(T* tp, Buffer& b) {
      using alloc_type =
          typename std::allocator_traits<allocator_type>::template rebind_alloc<T>;
      alloc_type a(b.alloc); // rebind allocator
      detail::destroy_buffer(a, tp, b.size);
      b.ptr = nullptr;
    }
  };

  struct setter {
    template <class T, class OBuffer, class Buffer>
    void operator()(T* optr, const OBuffer& ob, Buffer& b) {
      if (b.size == ob.size && b.type == ob.type) {
        std::copy(optr, optr + ob.size, reinterpret_cast<T*>(b.ptr));
      } else {
        apply(destroyer(), b);
        b.size = ob.size;
        b.set(b.template create<T>(optr));
      }
    }

    template <class T, class Buffer, class U>
    void operator()(T* tp, Buffer& b, std::size_t i, const U& u) {
      tp[i] = 0;
      adder()(tp, b, i, u);
    }
  };

  struct incrementor {
    template <class T, class Buffer>
    void operator()(T* tp, Buffer& b, std::size_t i) {
      if (!detail::safe_increment(tp[i])) {
        using U = mp11::mp_at_c<types, (type_index<T>() + 1)>;
        U* ptr = b.template create<U>(tp);
        destroyer()(tp, b);
        b.set(ptr);
        ++reinterpret_cast<U*>(b.ptr)[i];
      }
    }

    template <class Buffer>
    void operator()(mp_int* tp, Buffer&, std::size_t i) {
      ++tp[i];
    }

    template <class Buffer>
    void operator()(double* tp, Buffer&, std::size_t i) {
      ++tp[i];
    }
  };

  struct adder {
    template <class Buffer, class U>
    void if_U_is_integral(std::true_type, mp_int* tp, Buffer&, std::size_t i,
                          const U& x) {
      tp[i] += x;
    }

    template <class T, class Buffer, class U>
    void if_U_is_integral(std::true_type, T* tp, Buffer& b, std::size_t i, const U& x) {
      if (detail::safe_radd(tp[i], x)) return;
      if (x >= 0) {
        using V = mp11::mp_at_c<types, (type_index<T>() + 1)>;
        auto ptr = b.template create<V>(tp);
        destroyer()(tp, b);
        b.set(ptr);
        if_U_is_integral(std::true_type{}, static_cast<V*>(b.ptr), b, i, x);
      } else {
        if_U_is_integral(std::false_type{}, tp, b, i, x);
      }
    }

    template <class T, class Buffer, class U>
    void if_U_is_integral(std::false_type, T* tp, Buffer& b, std::size_t i, const U& x) {
      auto ptr = b.template create<double>(tp);
      destroyer()(tp, b);
      b.set(ptr);
      operator()(static_cast<double*>(b.ptr), b, i, x);
    }

    template <class T, class Buffer, class U>
    void operator()(T* tp, Buffer& b, std::size_t i, const U& x) {
      if_U_is_integral(std::is_integral<U>{}, tp, b, i, x);
    }

    template <class Buffer, class U>
    void operator()(double* tp, Buffer&, std::size_t i, const U& x) {
      tp[i] += static_cast<double>(x);
    }
  };

  struct buffer_adder {
    template <class T, class OBuffer, class Buffer>
    void operator()(T* tp, const OBuffer&, Buffer& b) {
      for (std::size_t i = 0; i < b.size; ++i) { apply(adder(), b, i, tp[i]); }
    }
  };

  struct buffer_subtractor {
    template <class T, class OBuffer, class Buffer>
    void operator()(T* tp, const OBuffer&, Buffer& b) {
      for (std::size_t i = 0; i < b.size; ++i) { apply(adder(), b, i, negate(tp[i])); }
    }
  };

  struct getter {
    template <class T, class Buffer>
    double operator()(T* tp, Buffer&, std::size_t i) {
      return static_cast<double>(tp[i]);
    }
  };

  struct multiplier {
    template <class T, class Buffer>
    void operator()(T* tp, Buffer& b, const double x) {
      // potential lossy conversion that cannot be avoided
      auto ptr = b.template create<double>(tp);
      destroyer()(tp, b);
      b.set(ptr);
      operator()(reinterpret_cast<double*>(b.ptr), b, x);
    }

    template <class Buffer>
    void operator()(double* tp, Buffer& b, const double x) {
      for (auto end = tp + b.size; tp != end; ++tp) *tp *= x;
    }
  };

  buffer_type buffer;
};
} // namespace histogram
} // namespace boost

#endif
