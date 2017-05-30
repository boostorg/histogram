// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP_
#define _BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP_

#include <algorithm>
#include <boost/cstdint.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/weight.hpp>
#include <boost/mpl/int.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/variant.hpp>
#include <limits>
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

using mp_int = multiprecision::cpp_int;

template <template <class> class Allocator, typename T>
class array : public Allocator<T> {
public:
  using value_type = T;
  array(const std::size_t &s) : size(s) {
    create();
    std::fill(begin(), end(), T(0));
  }
  array() = default;
  array(const array &rhs) : Allocator<T>(rhs), size(rhs.size) {
    create();
    std::copy(rhs.begin(), rhs.end(), begin());
  }
  array &operator=(const array &rhs) {
    if (this != &rhs) {
      if (size != rhs.size) {
        destroy();
        Allocator<T>::operator=(rhs);
        size = rhs.size;
        create();
      }
      std::copy(rhs.begin(), rhs.end(), begin());
    }
    return *this;
  }
  array(array &&rhs)
      : Allocator<T>(std::move(rhs)), size(rhs.size), ptr(rhs.ptr) {
    rhs.size = 0;
  }
  array &operator=(array &&rhs) {
    if (this != &rhs) {
      Allocator<T>::operator=(std::move(rhs));
      size = rhs.size;
      ptr = rhs.ptr;
      rhs.size = 0;
    }
    return *this;
  }
  ~array() { destroy(); }

  template <typename U>
  array(const array<Allocator, U> &rhs,
        std::size_t n = std::numeric_limits<std::size_t>::max())
      : Allocator<T>(rhs), size(rhs.size) {
    create();
    std::copy(rhs.begin(), rhs.begin() + std::min(n, size), begin());
  }

  T &operator[](const std::size_t i) { return ptr[i]; }
  const T &operator[](const std::size_t i) const { return ptr[i]; }

  T *begin() { return ptr; }
  T *end() { return ptr + size; }
  const T *begin() const { return ptr; }
  const T *end() const { return ptr + size; }

  std::size_t size = 0;

private:
  void create() {
    if (size) {
      ptr = Allocator<T>::allocate(size);
      if (!std::is_pod<T>::value) {
        new (ptr) T[size];
      }
    }
  }
  void destroy() {
    if (size) {
      if (!std::is_pod<T>::value) {
        for (auto it = begin(); it != end(); ++it) {
          it->~T();
        }
      }
      Allocator<T>::deallocate(ptr, size);
    }
  }
  T *ptr = nullptr;
};

template <template <class> class Allocator> class array<Allocator, void> {
public:
  using value_type = void;
  array(const std::size_t &s) : size(s) {}
  array() = default;
  array(const array &) = default;
  array &operator=(const array &) = default;
  array(array &&) = default;
  array &operator=(array &&) = default;
  std::size_t size = 0;
};

template <typename T> struct next_type;
template <> struct next_type<uint8_t> { using type = uint16_t; };
template <> struct next_type<uint16_t> { using type = uint32_t; };
template <> struct next_type<uint32_t> { using type = uint64_t; };
template <> struct next_type<uint64_t> { using type = mp_int; };

} // namespace detail

template <template <class> class Allocator = std::allocator>
class adaptive_storage {
  template <typename T> using array = detail::array<Allocator, T>;
  template <typename T> using next = typename detail::next_type<T>::type;
  using weight = detail::weight;
  using mp_int = detail::mp_int;
  using buffer_type =
      variant<array<void>, array<uint8_t>, array<uint16_t>, array<uint32_t>,
              array<uint64_t>, array<mp_int>, array<weight>>;

public:
  using value_type = double;

  explicit adaptive_storage(std::size_t s) : buffer_(array<void>(s)) {}

  adaptive_storage() = default;
  adaptive_storage(const adaptive_storage &) = default;
  adaptive_storage &operator=(const adaptive_storage &) = default;
  adaptive_storage(adaptive_storage &&) = default;
  adaptive_storage &operator=(adaptive_storage &&) = default;

  template <typename S, typename = detail::is_storage<S>>
  explicit adaptive_storage(const S &rhs) : buffer_(array<void>(rhs.size())) {
    for (std::size_t i = 0, n = rhs.size(); i < n; ++i) {
      apply_visitor(
          assign_visitor<typename S::value_type>(i, rhs.value(i), buffer_),
          buffer_);
    }
  }

  template <typename S> adaptive_storage &operator=(const S &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      if (size() != rhs.size()) {
        buffer_ = array<void>(rhs.size());
      }
      for (std::size_t i = 0, n = rhs.size(); i < n; ++i) {
        apply_visitor(
            assign_visitor<typename S::value_type>(i, rhs.value(i), buffer_),
            buffer_);
      }
    }
    return *this;
  }

  std::size_t size() const { return apply_visitor(size_visitor(), buffer_); }

  void increase(std::size_t i) {
    apply_visitor(increase_visitor(i, buffer_), buffer_);
  }

  template <typename Value> void increase(std::size_t i, const Value &n) {
    apply_visitor(add_visitor<Value>(i, n, buffer_), buffer_);
  }

  void weighted_increase(std::size_t i, value_type weight) {
    apply_visitor(wincrease_visitor(i, weight, buffer_), buffer_);
  }

  void add(std::size_t i, const value_type &val, const value_type &var) {
    if (val == var) {
      apply_visitor(add_visitor<value_type>(i, val, buffer_), buffer_);
    } else {
      if (!boost::get<array<weight>>(&buffer_)) {
        apply_visitor(wincrease_visitor(0, 0.0, buffer_), buffer_);
      }
      auto &b = boost::get<array<weight>>(buffer_);
      b[i].w += val;
      b[i].w2 += var;
    }
  }

  value_type value(std::size_t i) const {
    return apply_visitor(value_visitor(i), buffer_);
  }

  value_type variance(std::size_t i) const {
    return apply_visitor(variance_visitor(i), buffer_);
  }

  bool operator==(const adaptive_storage &rhs) const {
    return apply_visitor(bicmp_visitor(), buffer_, rhs.buffer_);
  }

  template <typename S> bool operator==(const S &rhs) const {
    return apply_visitor(cmp_visitor<S>(rhs), buffer_);
  }

private:
  struct size_visitor : public static_visitor<std::size_t> {
    template <typename Array> std::size_t operator()(const Array &b) const {
      return b.size;
    }
  };

  template <typename Value>
  struct assign_visitor : public static_visitor<void> {
    const std::size_t &idx;
    const Value &value;
    buffer_type &buffer;
    assign_visitor(const std::size_t &i, const Value &v, buffer_type &b)
        : idx(i), value(v), buffer(b) {}

    template <typename Array> void operator()(Array &b) const {
      using T = typename Array::value_type;
      if (value <= std::numeric_limits<T>::max()) {
        b[idx] = value;
      } else {
        buffer = array<next<T>>(b, idx);
        (*this)(get<array<next<T>>>(buffer));
      }
    }

    void operator()(array<void> &b) const {
      buffer = array<uint8_t>(b.size);
      (*this)(get<array<uint8_t>>(buffer));
    }

    void operator()(array<mp_int> &b) const {
      b[idx] = static_cast<mp_int>(value);
    }

    void operator()(array<weight> &b) const { b[idx] = value; }
  };

  struct increase_visitor : public static_visitor<void> {
    const std::size_t &idx;
    buffer_type &buffer;
    increase_visitor(const std::size_t &i, buffer_type &b)
        : idx(i), buffer(b) {}

    template <typename Array> void operator()(Array &b) const {
      using T = typename Array::value_type;
      T &x = b[idx];
      if (x < std::numeric_limits<T>::max()) {
        ++x;
      } else {
        array<next<T>> nb = b;
        ++nb[idx];
        buffer = std::move(nb);
      }
    }

    void operator()(array<void> &b) const {
      array<uint8_t> nb(b.size);
      ++nb[idx];
      buffer = std::move(nb);
    }

    void operator()(array<mp_int> &b) const { ++b[idx]; }

    void operator()(array<weight> &b) const { ++b[idx]; }
  };

  struct wincrease_visitor : public static_visitor<void> {
    const std::size_t &idx;
    const value_type &w;
    buffer_type &buffer;
    wincrease_visitor(const std::size_t &i, const value_type &v, buffer_type &b)
        : idx(i), w(v), buffer(b) {}

    template <typename Array> void operator()(Array &b) const {
      array<weight> nb(b);
      nb[idx].add_weight(w);
      buffer = std::move(nb);
    }

    void operator()(array<void> &b) const {
      array<weight> nb(b.size);
      nb[idx].add_weight(w);
      buffer = std::move(nb);
    }

    void operator()(array<weight> &b) const { b[idx].add_weight(w); }
  };

  struct value_visitor : public static_visitor<value_type> {
    const std::size_t &idx;
    value_visitor(const std::size_t &i) : idx(i) {}

    template <typename Array> value_type operator()(const Array &b) const {
      return static_cast<value_type>(b[idx]);
    }

    value_type operator()(const array<void> & /*b*/) const {
      return value_type(0);
    }

    value_type operator()(const array<weight> &b) const {
      return static_cast<value_type>(b[idx].w);
    }
  };

  struct variance_visitor : public static_visitor<value_type> {
    const std::size_t &idx;
    variance_visitor(const std::size_t &i) : idx(i) {}

    template <typename Array> value_type operator()(const Array &b) const {
      return static_cast<value_type>(b[idx]);
    }

    value_type operator()(const array<void> & /*b*/) const {
      return value_type(0);
    }

    value_type operator()(const array<weight> &b) const {
      return static_cast<value_type>(b[idx].w2);
    }
  };

  template <typename Value> struct add_visitor : public static_visitor<void> {
    const std::size_t &idx;
    const Value &value;
    buffer_type &buffer;
    add_visitor(const std::size_t &i, const Value &v, buffer_type &b)
        : idx(i), value(v), buffer(b) {}

    template <typename Array> void operator()(Array &b) const {
      using T = typename Array::value_type;
      T &x = b[idx];
      if (static_cast<T>(std::numeric_limits<T>::max() - x) > value) {
        x += value;
      } else {
        buffer = array<next<T>>(b);
        (*this)(get<array<next<T>>>(buffer));
      }
    }

    void operator()(array<void> &b) const {
      if (value > 0) {
        buffer = array<uint8_t>(b.size);
        operator()(get<array<uint8_t>>(buffer));
      }
    }

    void operator()(array<mp_int> &b) const {
      b[idx] += static_cast<mp_int>(value);
    }

    void operator()(array<weight> &b) const { b[idx] += value; }
  };

  struct bicmp_visitor : public static_visitor<bool> {
    template <typename Array1, typename Array2>
    bool operator()(const Array1 &b1, const Array2 &b2) const {
      if (b1.size != b2.size)
        return false;
      for (std::size_t i = 0; i < b1.size; ++i) {
        if (b1[i] != b2[i])
          return false;
      }
      return true;
    }

    template <typename Array>
    bool operator()(const Array &b1, const array<void> &b2) const {
      if (b1.size != b2.size)
        return false;
      for (std::size_t i = 0; i < b1.size; ++i) {
        if (b1[i] != 0)
          return false;
      }
      return true;
    }

    template <typename Array>
    bool operator()(const array<void> &b1, const Array &b2) const {
      if (b1.size != b2.size)
        return false;
      for (std::size_t i = 0; i < b1.size; ++i) {
        if (b2[i] != 0)
          return false;
      }
      return true;
    }

    bool operator()(const array<void> &b1, const array<void> &b2) const {
      return b1.size == b2.size;
    }
  };

  template <typename S> struct cmp_visitor : public static_visitor<bool> {
    const S &os;
    cmp_visitor(const S &o) : os(o) {}

    template <typename Array> bool operator()(const Array &b) const {
      if (b.size != os.size())
        return false;
      for (std::size_t i = 0; i < b.size; ++i) {
        if (b[i] != os.value(i))
          return false;
      }
      return true;
    }

    bool operator()(const array<mp_int> &b) const {
      if (b.size != os.size())
        return false;
      for (std::size_t i = 0; i < b.size; ++i) {
        if (static_cast<value_type>(b[i]) != os.value(i))
          return false;
      }
      return true;
    }

    bool operator()(const array<weight> &b) const {
      if (b.size != os.size())
        return false;
      for (std::size_t i = 0; i < b.size; ++i) {
        auto &x = b[i];
        if (x.w != os.value(i) || x.w2 != os.variance(i))
          return false;
      }
      return true;
    }

    bool operator()(const array<void> &b) const {
      if (b.size != os.size())
        return false;
      for (std::size_t i = 0; i < b.size; ++i) {
        if (os.value(i) != 0)
          return false;
      }
      return true;
    }
  };

  buffer_type buffer_;

  friend class ::boost::python::access;
  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

template <template <class> class Allocator, typename S>
bool operator==(const S &a, const adaptive_storage<Allocator> &b) {
  return b == a;
}

} // namespace histogram
} // namespace boost

#endif
