// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UNLIMTED_STORAGE_HPP
#define BOOST_HISTOGRAM_UNLIMTED_STORAGE_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config/workaround.hpp>
#include <boost/cstdint.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

namespace boost {
namespace histogram {

namespace detail {

template <class Allocator>
struct mp_int;

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
  if (static_cast<T>(std::numeric_limits<T>::max() - t) >= u) {
    t += static_cast<T>(u); // static_cast to suppress conversion warning
    return true;
  }
  return false;
}

template <class T, class U>
bool safe_radd(std::false_type, T& t, const U& u) {
  static_assert(std::is_integral<U>::value, "U must be integral type");
  if (u >= 0) {
    if (static_cast<T>(std::numeric_limits<T>::max() - t) >=
        static_cast<std::make_unsigned_t<U>>(u)) {
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

template <class T, class A>
bool safe_radd(std::false_type, T& t, const mp_int<A>& u) {
  if (u >= 0) {
    if (std::numeric_limits<T>::max() - t > u) {
      t += static_cast<T>(u);
      return true;
    }
  }
  if (u + t >= 0) {
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

// use boost.multiprecision.cpp_int in your own code, it is much more sophisticated
// than this implementation; we use it here to reduce coupling between boost libs
template <class Allocator>
struct mp_int {
  explicit mp_int(Allocator a = {}) : data(1, 0, std::move(a)) {}
  explicit mp_int(uint64_t v, Allocator a = {}) : data(1, v, std::move(a)) {}
  mp_int(const mp_int&) = default;
  mp_int& operator=(const mp_int&) = default;
  mp_int(mp_int&&) = default;
  mp_int& operator=(mp_int&&) = default;

  mp_int& operator=(uint64_t o) {
    data = decltype(data)(1, o);
    return *this;
  }

  mp_int& operator++() {
    BOOST_ASSERT(data.size() >= 1);
    std::size_t i = 0;
    while (!safe_increment(data[i])) {
      data[i] = 0;
      ++i;
      if (i == data.size()) {
        data.push_back(1);
        break;
      }
    }
    return *this;
  }

  mp_int& operator+=(const mp_int& o) {
    if (this == &o) {
      auto tmp{o};
      return operator+=(tmp);
    }
    bool carry = false;
    std::size_t i = 0;
    for (uint64_t oi : o.data) {
      auto& di = maybe_extend(i);
      if (carry) {
        if (safe_increment(oi))
          carry = false;
        else {
          ++i;
          continue;
        }
      }
      if (!safe_radd(di, oi)) {
        add_remainder(di, oi);
        carry = true;
      }
      ++i;
    }
    while (carry) {
      auto& di = maybe_extend(i);
      if (safe_increment(di)) break;
      di = 0;
      ++i;
    }
    return *this;
  }

  mp_int& operator+=(uint64_t o) {
    BOOST_ASSERT(data.size() >= 1);
    if (safe_radd(data[0], o)) return *this;
    add_remainder(data[0], o);
    // carry the one, data may grow several times
    std::size_t i = 1;
    while (true) {
      auto& di = maybe_extend(i);
      if (safe_increment(di)) break;
      di = 0;
      ++i;
    }
    return *this;
  }

  operator double() const noexcept {
    BOOST_ASSERT(data.size() >= 1);
    double result = data[0];
    std::size_t i = 0;
    while (++i < data.size())
      result += static_cast<double>(data[i]) * std::pow(2.0, i * 64);
    return result;
  }

  bool operator<(uint64_t o) const noexcept {
    BOOST_ASSERT(data.size() >= 1);
    return data.size() == 1 && data[0] < o;
  }

  bool operator>(uint64_t o) const noexcept {
    BOOST_ASSERT(data.size() >= 1);
    return data.size() > 1 || data[0] > o;
  }

  bool operator==(uint64_t o) const noexcept {
    BOOST_ASSERT(data.size() >= 1);
    return data.size() == 1 && data[0] == o;
  }

  bool operator<(double o) const noexcept { return operator double() < o; }

  bool operator>(double o) const noexcept { return operator double() > o; }

  bool operator==(double o) const noexcept { return operator double() == o; }

  bool operator<(const mp_int& o) const noexcept {
    if (data.size() < o.data.size()) return true;
    if (data.size() > o.data.size()) return false;
    auto s = data.size();
    while (s > 0) {
      --s;
      if (data[s] < o.data[s]) return true;
      if (data[s] > o.data[s]) return false;
    }
    return false; // args are equal
  }

  bool operator>(const mp_int& o) const noexcept {
    if (data.size() > o.data.size()) return true;
    if (data.size() < o.data.size()) return false;
    auto s = data.size();
    while (s > 0) {
      --s;
      if (data[s] > o.data[s]) return true;
      if (data[s] < o.data[s]) return false;
    }
    return false; // args are equal
  }

  bool operator==(const mp_int& o) const noexcept {
    if (data.size() != o.data.size()) return false;
    for (std::size_t s = 0; s < data.size(); ++s)
      if (data[s] != o.data[s]) return false;
    return true; // args are equal
  }

  template <class T>
  bool operator<(const T& o) const noexcept {
    return std::is_integral<T>::value ? operator<(static_cast<uint64_t>(o)) :
                                      operator<(static_cast<double>(o));
  }

  template <class T>
  bool operator>(const T& o) const noexcept {
    return std::is_integral<T>::value ? operator>(static_cast<uint64_t>(o)) :
                                      operator>(static_cast<double>(o));
  }

  template <class T>
  bool operator==(const T& o) const noexcept {
    return std::is_integral<T>::value ? operator==(static_cast<uint64_t>(o)) :
                                      operator==(static_cast<double>(o));
  }

  template <class T>
  bool operator<=(const T& o) const noexcept {
    return !operator>(o);
  }

  template <class T>
  bool operator>=(const T& o) const noexcept {
    return !operator<(o);
  }

  template <class T>
  bool operator!=(const T& o) const noexcept {
    return !operator==(o);
  }

  template <class T>
  friend bool operator<(const T& a, const mp_int& b) noexcept {
    return !(b >= a);
  }

  template <class T>
  friend bool operator>(const T& a, const mp_int& b) noexcept {
    return !(b <= a);
  }

  template <class T>
  friend bool operator<=(const T& a, const mp_int& b) noexcept {
    return !(b > a);
  }

  template <class T>
  friend bool operator>=(const T& a, const mp_int& b) noexcept {
    return !(b < a);
  }

  template <class T>
  friend bool operator==(const T& a, const mp_int& b) noexcept {
    return b == a;
  }

  template <class T>
  friend bool operator!=(const T& a, const mp_int& b) noexcept {
    return b != a;
  }

  uint64_t& maybe_extend(std::size_t i) {
    while (i >= data.size()) data.push_back(0);
    return data[i];
  }

  static void add_remainder(uint64_t& d, const uint64_t o) noexcept {
    BOOST_ASSERT(d > 0);
    // in decimal system it would look like this:
    // 8 + 8 = 6 = 8 - (9 - 8) - 1
    // 9 + 1 = 0 = 9 - (9 - 1) - 1
    auto tmp = std::numeric_limits<uint64_t>::max();
    tmp -= o;
    --d -= tmp;
  }

  std::vector<uint64_t, Allocator> data;
};

template <class Allocator>
auto create_buffer(Allocator& a, std::size_t n) {
  using AT = std::allocator_traits<Allocator>;
  auto ptr = AT::allocate(a, n); // may throw
  static_assert(std::is_trivially_copyable<decltype(ptr)>::value,
                "ptr must be trivially copyable");
  auto it = ptr;
  const auto end = ptr + n;
  try {
    // this loop may throw
    while (it != end) AT::construct(a, it++, typename AT::value_type{});
  } catch (...) {
    // release resources that were already acquired before rethrowing
    while (it != ptr) AT::destroy(a, --it);
    AT::deallocate(a, ptr, n);
    throw;
  }
  return ptr;
}

template <class Allocator, class Iterator>
auto create_buffer(Allocator& a, std::size_t n, Iterator iter) {
  BOOST_ASSERT(n > 0);
  using AT = std::allocator_traits<Allocator>;
  auto ptr = AT::allocate(a, n); // may throw
  static_assert(std::is_trivially_copyable<decltype(ptr)>::value,
                "ptr must be trivially copyable");
  auto it = ptr;
  const auto end = ptr + n;
  try {
    // this loop may throw
    while (it != end) AT::construct(a, it++, *iter++);
  } catch (...) {
    // release resources that were already acquired before rethrowing
    while (it != ptr) AT::destroy(a, --it);
    AT::deallocate(a, ptr, n);
    throw;
  }
  return ptr;
}

template <class Allocator>
void destroy_buffer(Allocator& a, typename std::allocator_traits<Allocator>::pointer p,
                    std::size_t n) {
  BOOST_ASSERT(p);
  BOOST_ASSERT(n > 0);
  using AT = std::allocator_traits<Allocator>;
  auto it = p + n;
  while (it != p) AT::destroy(a, --it);
  AT::deallocate(a, p, n);
}

} // namespace detail

/**
  Memory-efficient storage for integral counters which cannot overflow.

  This storage provides a no-overflow-guarantee if it is filled with integral weights
  only. This storage implementation keeps a contiguous array of elemental counters, one
  for each cell. If an operation is requested, which would overflow a counter, the whole
  array is replaced with another of a wider integral type, then the operation is executed.
  The storage uses integers of 8, 16, 32, 64 bits, and then switches to a multiprecision
  integral type, similar to those in
  [Boost.Multiprecision](https://www.boost.org/doc/libs/develop/libs/multiprecision/doc/html/index.html).

  A scaling operation or adding a floating point number turns the elements into doubles,
  which voids the no-overflow-guarantee.
*/
template <class Allocator>
class unlimited_storage {
  static_assert(
      std::is_same<typename std::allocator_traits<Allocator>::pointer,
                   typename std::allocator_traits<Allocator>::value_type*>::value,
      "unlimited_storage requires allocator with trivial pointer type");

public:
  struct storage_tag {};
  using allocator_type = Allocator;
  using value_type = double;
  using mp_int = detail::mp_int<
      typename std::allocator_traits<allocator_type>::template rebind_alloc<uint64_t>>;

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
  static constexpr char type_index() noexcept {
    return static_cast<char>(mp11::mp_find<types, T>::value);
  }

  struct buffer_type {
    allocator_type alloc;
    std::size_t size = 0;
    char type = 0;
    void* ptr = nullptr;

    template <class F, class... Ts>
    decltype(auto) apply(F&& f, Ts&&... ts) const {
      // this is intentionally not a switch, the if-chain is faster in benchmarks
      if (type == type_index<uint8_t>())
        return f(static_cast<uint8_t*>(ptr), std::forward<Ts>(ts)...);
      if (type == type_index<uint16_t>())
        return f(static_cast<uint16_t*>(ptr), std::forward<Ts>(ts)...);
      if (type == type_index<uint32_t>())
        return f(static_cast<uint32_t*>(ptr), std::forward<Ts>(ts)...);
      if (type == type_index<uint64_t>())
        return f(static_cast<uint64_t*>(ptr), std::forward<Ts>(ts)...);
      if (type == type_index<mp_int>())
        return f(static_cast<mp_int*>(ptr), std::forward<Ts>(ts)...);
      return f(static_cast<double*>(ptr), std::forward<Ts>(ts)...);
    }

    buffer_type(std::size_t s = 0, allocator_type a = {})
        : alloc(std::move(a)), size(s), type(type_index<uint8_t>()) {
      if (size > 0) {
        // rebind allocator
        using alloc_type = typename std::allocator_traits<
            allocator_type>::template rebind_alloc<uint8_t>;
        alloc_type a(alloc);
        try {
          ptr = detail::create_buffer(a, size); // may throw
        } catch (...) {
          size = 0;
          throw;
        }
      }
    }

    buffer_type(buffer_type&& o) noexcept
        : alloc(std::move(o.alloc)), size(o.size), type(o.type), ptr(o.ptr) {
      o.size = 0;
      o.type = 0;
      o.ptr = nullptr;
    }

    buffer_type& operator=(buffer_type&& o) noexcept {
      if (this != &o) {
        using std::swap;
        swap(alloc, o.alloc);
        swap(size, o.size);
        swap(type, o.type);
        swap(ptr, o.ptr);
      }
      return *this;
    }

    buffer_type(const buffer_type& o) : alloc(o.alloc) {
      o.apply([this, &o](auto* otp) {
        using T = detail::remove_cvref_t<decltype(*otp)>;
        this->template make<T>(o.size, otp);
      });
    }

    buffer_type& operator=(const buffer_type& o) {
      *this = buffer_type(o);
      return *this;
    }

    ~buffer_type() noexcept { destroy(); }

    void destroy() noexcept {
      BOOST_ASSERT((ptr == nullptr) == (size == 0));
      if (ptr == nullptr) return;
      apply([this](auto* tp) {
        using T = detail::remove_cvref_t<decltype(*tp)>;
        using alloc_type =
            typename std::allocator_traits<allocator_type>::template rebind_alloc<T>;
        alloc_type a(alloc); // rebind allocator
        detail::destroy_buffer(a, tp, size);
      });
      size = 0;
      type = 0;
      ptr = nullptr;
    }

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable : 4244) // possible loss of data
#endif

    template <class T>
    void make(std::size_t n) {
      // note: order of commands is to not leave buffer in invalid state upon throw
      destroy();
      if (n > 0) {
        // rebind allocator
        using alloc_type =
            typename std::allocator_traits<allocator_type>::template rebind_alloc<T>;
        alloc_type a(alloc);
        ptr = detail::create_buffer(a, n); // may throw
      }
      size = n;
      type = type_index<T>();
    }

    template <class T, class U>
    void make(std::size_t n, U iter) {
      // note: iter may be current ptr, so create new buffer before deleting old buffer
      T* new_ptr = nullptr;
      const auto new_type = type_index<T>();
      if (n > 0) {
        // rebind allocator
        using alloc_type =
            typename std::allocator_traits<allocator_type>::template rebind_alloc<T>;
        alloc_type a(alloc);
        new_ptr = detail::create_buffer(a, n, iter); // may throw
      }
      destroy();
      size = n;
      type = new_type;
      ptr = new_ptr;
    }

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif
  };

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

    operator double() const {
      return buffer_->apply(
          [this](const auto* tp) { return static_cast<double>(tp[idx_]); });
    }

  protected:
    template <class Binary, class U>
    bool op(const reference_t<U>& rhs) const {
      const auto i = idx_;
      const auto j = rhs.idx_;
      return buffer_->apply([i, j, &rhs](const auto* ptr) {
        const auto& pi = ptr[i];
        return rhs.buffer_->apply([&pi, j](const auto* q) { return Binary()(pi, q[j]); });
      });
    }

    template <class Binary, class U>
    bool op(const U& rhs) const {
      const auto i = idx_;
      return buffer_->apply([i, &rhs](const auto* tp) { return Binary()(tp[i], rhs); });
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
      t.buffer_->apply([this, &t](const auto* otp) { *this = otp[t.idx_]; });
      return *this;
    }

    template <class T>
    reference& operator=(const T& t) {
      base_type::buffer_->apply([this, &t](auto* tp) {
        tp[this->idx_] = 0;
        adder()(tp, *(this->buffer_), this->idx_, t);
      });
      return *this;
    }

    template <class T>
    reference& operator+=(const T& t) {
      base_type::buffer_->apply(adder(), *base_type::buffer_, base_type::idx_, t);
      return *this;
    }

    template <class T>
    reference& operator-=(const T& t) {
      base_type::buffer_->apply(adder(), *base_type::buffer_, base_type::idx_,
                                -static_cast<double>(t));
      return *this;
    }

    template <class T>
    reference& operator*=(const T& t) {
      base_type::buffer_->apply(multiplier(), *base_type::buffer_, base_type::idx_, t);
      return *this;
    }

    template <class T>
    reference& operator/=(const T& t) {
      base_type::buffer_->apply(multiplier(), *base_type::buffer_, base_type::idx_,
                                1.0 / static_cast<double>(t));
      return *this;
    }

    reference& operator++() {
      base_type::buffer_->apply(incrementor(), *base_type::buffer_, base_type::idx_);
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

  explicit unlimited_storage(allocator_type a = {}) : buffer(0, std::move(a)) {}
  unlimited_storage(const unlimited_storage&) = default;
  unlimited_storage& operator=(const unlimited_storage&) = default;
  unlimited_storage(unlimited_storage&&) = default;
  unlimited_storage& operator=(unlimited_storage&&) = default;

  template <class T>
  unlimited_storage(const storage_adaptor<T>& s) {
    using V = detail::remove_cvref_t<decltype(s[0])>;
    constexpr auto ti = type_index<V>();
    if (ti < mp11::mp_size<types>::value)
      buffer.template make<V>(s.size(), s.begin());
    else {
      buffer.template make<double>(s.size(), s.begin());
    }
  }

  template <class Iterable, class = detail::requires_iterable<Iterable>>
  unlimited_storage& operator=(const Iterable& s) {
    *this = unlimited_storage(s);
    return *this;
  }

  allocator_type get_allocator() const { return buffer.alloc; }

  void reset(std::size_t s) { buffer.template make<uint8_t>(s); }

  std::size_t size() const noexcept { return buffer.size; }

  reference operator[](std::size_t i) noexcept { return {&buffer, i}; }
  const_reference operator[](std::size_t i) const noexcept { return {&buffer, i}; }

  bool operator==(const unlimited_storage& o) const noexcept {
    if (size() != o.size()) return false;
    return buffer.apply([&o](const auto* ptr) {
      return o.buffer.apply([ptr, &o](const auto* optr) {
        return std::equal(ptr, ptr + o.size(), optr, equal_to());
      });
    });
  }

  template <class T>
  bool operator==(const T& o) const {
    if (size() != o.size()) return false;
    return buffer.apply([&o](const auto* ptr) {
      return std::equal(ptr, ptr + o.size(), std::begin(o), equal_to());
    });
  }

  unlimited_storage& operator*=(const double x) {
    buffer.apply(multiplier(), buffer, x);
    return *this;
  }

  iterator begin() noexcept { return {&buffer, 0}; }
  iterator end() noexcept { return {&buffer, size()}; }
  const_iterator begin() const noexcept { return {&buffer, 0}; }
  const_iterator end() const noexcept { return {&buffer, size()}; }

  /// @private used by unit tests, not part of generic storage interface
  template <class T>
  unlimited_storage(std::size_t s, const T* p, allocator_type a = {})
      : buffer(0, std::move(a)) {
    buffer.template make<T>(s, p);
  }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  struct incrementor {
    template <class T, class Buffer>
    void operator()(T* tp, Buffer& b, std::size_t i) {
      if (!detail::safe_increment(tp[i])) {
        using U = mp11::mp_at_c<types, (type_index<T>() + 1)>;
        b.template make<U>(b.size, tp);
        ++static_cast<U*>(b.ptr)[i];
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
        b.template make<V>(b.size, tp);
        if_U_is_integral(std::true_type{}, static_cast<V*>(b.ptr), b, i, x);
      } else {
        if_U_is_integral(std::false_type{}, tp, b, i, x);
      }
    }

    template <class T, class Buffer, class U>
    void if_U_is_integral(std::false_type, T* tp, Buffer& b, std::size_t i, const U& x) {
      b.template make<double>(b.size, tp);
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

  struct multiplier {
    template <class T, class Buffer>
    void operator()(T* tp, Buffer& b, const double x) {
      // potential lossy conversion that cannot be avoided
      b.template make<double>(b.size, tp);
      operator()(static_cast<double*>(b.ptr), b, x);
    }

    template <class Buffer>
    void operator()(double* tp, Buffer& b, const double x) {
      for (auto end = tp + b.size; tp != end; ++tp) *tp *= x;
    }

    template <class T, class Buffer, class U>
    void operator()(T* tp, Buffer& b, std::size_t i, const U& x) {
      b.template make<double>(b.size, tp);
      operator()(static_cast<double*>(b.ptr), b, i, x);
    }

    template <class Buffer, class U>
    void operator()(double* tp, Buffer&, std::size_t i, const U& x) {
      tp[i] *= static_cast<double>(x);
    }
  };

  buffer_type buffer;
};

} // namespace histogram
} // namespace boost

#endif
