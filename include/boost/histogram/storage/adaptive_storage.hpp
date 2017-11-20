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
#include <boost/histogram/detail/weight_counter.hpp>
#include <boost/mpl/int.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/variant.hpp>
#include <limits>
#include <memory>
#include <type_traits>
#ifdef BOOST_HISTOGRAM_TRACE_ALLOCS
#include <iostream>
#include <boost/core/typeinfo.hpp>
#endif


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

template <typename T>
inline T* alloc(std::size_t s) {
#ifdef BOOST_HISTOGRAM_TRACE_ALLOCS
  boost::core::typeinfo const & ti = BOOST_CORE_TYPEID(T);
  std::cerr << "alloc " << boost::core::demangled_name( ti ) << "[" << s << "]" << std::endl;
#endif
  return new T[s];
}

class array_base {
public:
  explicit array_base(const std::size_t s) : size(s) {}
  array_base() = default;
  array_base(const array_base &) = default;
  array_base &operator=(const array_base &) = default;
  array_base(array_base &&rhs) : size(rhs.size) { rhs.size = 0; }
  array_base &operator=(array_base &&rhs) {
    if (this != &rhs) {
      size = rhs.size;
      rhs.size = 0;
    }
    return *this;
  }
  std::size_t size = 0;
};

template <typename T> class array : public array_base {
public:
  explicit array(const std::size_t s) : array_base(s), ptr(alloc<T>(s)) {
    std::fill(begin(), end(), T(0));
  }
  array() = default;
  array(const array &rhs) : array_base(rhs), ptr(alloc<T>(rhs.size)) {
    std::copy(rhs.begin(), rhs.end(), begin());
  }
  array &operator=(const array &rhs) {
    if (this != &rhs) {
      if (size != rhs.size) {
        size = rhs.size;
        ptr.reset(alloc<T>(size));
      }
      std::copy(rhs.begin(), rhs.end(), begin());
    }
    return *this;
  }
  array(array &&rhs) : array_base(std::move(rhs)), ptr(std::move(rhs.ptr)) {
    rhs.size = 0;
  }
  array &operator=(array &&rhs) {
    if (this != &rhs) {
      size = rhs.size;
      ptr = std::move(rhs.ptr);
      rhs.size = 0;
    }
    return *this;
  }

  // copy only up to nmax elements
  template <typename U>
  array(const array<U> &rhs,
        std::size_t nmax = std::numeric_limits<std::size_t>::max())
      : array_base(rhs), ptr(alloc<T>(rhs.size)) {
    std::copy(rhs.begin(), rhs.begin() + std::min(nmax, size), begin());
  }

  T &operator[](const std::size_t i) { return ptr[i]; }
  const T &operator[](const std::size_t i) const { return ptr[i]; }

  T *begin() { return ptr.get(); }
  T *end() { return ptr.get() + size; }
  const T *begin() const { return ptr.get(); }
  const T *end() const { return ptr.get() + size; }

private:
  std::unique_ptr<T[]> ptr;
};

template <> class array<void> : public array_base {
public:
  using array_base::array_base;
};

using any_array =
    variant<array<void>, array<uint8_t>, array<uint16_t>, array<uint32_t>,
            array<uint64_t>, array<mp_int>, array<weight_counter>>;

template <typename T> struct next_type;
template <> struct next_type<uint8_t> { using type = uint16_t; };
template <> struct next_type<uint16_t> { using type = uint32_t; };
template <> struct next_type<uint32_t> { using type = uint64_t; };
template <> struct next_type<uint64_t> { using type = mp_int; };
template <typename T> using next = typename next_type<T>::type;

template <typename T> inline bool safe_increase(T &t) {
  if (t == std::numeric_limits<T>::max())
    return false;
  ++t;
  return true;
}

template <typename T, typename U> inline bool safe_assign(T &t, const U &u) {
  if (std::numeric_limits<T>::max() < std::numeric_limits<U>::max() &&
      std::numeric_limits<T>::max() < u)
    return false;
  t = static_cast<T>(u);
  return true;
}

template <typename T, typename U> inline bool safe_radd(T &t, const U &u) {
  if (static_cast<T>(std::numeric_limits<T>::max() - t) < u)
    return false;
  t += static_cast<T>(u);
  return true;
}

// float rounding is a mess, the equal sign is necessary here
template <typename T> inline bool safe_radd(T &t, const double u) {
  if ((std::numeric_limits<T>::max() - t) <= u)
    return false;
  t += u;
  return true;
}

struct size_visitor : public static_visitor<std::size_t> {
  template <typename Array> std::size_t operator()(const Array &b) const {
    return b.size;
  }
};

template <typename RHS> struct assign_visitor : public static_visitor<void> {
  any_array &lhs_any;
  const std::size_t idx;
  const RHS &rhs;
  assign_visitor(any_array &a, const std::size_t i, const RHS &x)
      : lhs_any(a), idx(i), rhs(x) {}

  template <typename T> void operator()(array<T> &lhs) const {
    if (!safe_assign(lhs[idx], rhs)) {
      lhs_any = array<next<T>>(lhs, idx);
      operator()(get<array<next<T>>>(lhs_any));
    }
  }

  void operator()(array<void> &lhs) const {
    lhs_any = array<uint8_t>(lhs.size);
    operator()(get<array<uint8_t>>(lhs_any));
  }

  void operator()(array<mp_int> &lhs) const { lhs[idx].assign(rhs); }

  void operator()(array<weight_counter> &lhs) const { lhs[idx] = rhs; }
};

struct increase_visitor : public static_visitor<void> {
  any_array &lhs_any;
  const std::size_t idx;
  increase_visitor(any_array &a, const std::size_t i) : lhs_any(a), idx(i) {}

  template <typename T> void operator()(array<T> &lhs) const {
    if (!safe_increase(lhs[idx])) {
      array<next<T>> a = lhs;
      ++a[idx];
      lhs_any = std::move(a);
    }
  }

  void operator()(array<void> &lhs) const {
    array<uint8_t> a(lhs.size);
    ++a[idx];
    lhs_any = std::move(a);
  }

  void operator()(array<mp_int> &lhs) const { ++lhs[idx]; }

  void operator()(array<weight_counter> &lhs) const { ++lhs[idx]; }
};

struct wincrease_visitor : public static_visitor<void> {
  any_array &lhs_any;
  const std::size_t idx;
  const double rhs;
  wincrease_visitor(any_array &l, const std::size_t i, const double r)
      : lhs_any(l), idx(i), rhs(r) {}

  template <typename T> void operator()(array<T> &lhs) const {
    array<weight_counter> a(lhs);
    a[idx] += rhs;
    lhs_any = std::move(a);
  }

  void operator()(array<void> &lhs) const {
    array<weight_counter> a(lhs.size);
    a[idx] += rhs;
    lhs_any = std::move(a);
  }

  void operator()(array<weight_counter> &lhs) const { lhs[idx] += rhs; }
};

struct value_visitor : public static_visitor<double> {
  const std::size_t idx;
  value_visitor(const std::size_t i) : idx(i) {}

  template <typename Array> double operator()(const Array &b) const {
    return static_cast<double>(b[idx]);
  }

  double operator()(const array<void> & /*b*/) const { return 0; }

  double operator()(const array<weight_counter> &b) const { return b[idx].w; }
};

struct variance_visitor : public static_visitor<double> {
  const std::size_t idx;
  variance_visitor(const std::size_t i) : idx(i) {}

  template <typename Array> double operator()(const Array &b) const {
    return static_cast<double>(b[idx]);
  }

  double operator()(const array<void> & /*b*/) const { return 0; }

  double operator()(const array<weight_counter> &b) const { return b[idx].w2; }
};

template <typename RHS> struct radd_visitor : public static_visitor<void> {
  any_array &lhs_any;
  const std::size_t idx;
  const RHS &rhs;
  radd_visitor(any_array &l, const std::size_t i, const RHS &r)
      : lhs_any(l), idx(i), rhs(r) {}

  template <typename T> void operator()(array<T> &lhs) const {
    if (!safe_radd(lhs[idx], rhs)) {
      lhs_any = array<next<T>>(lhs);
      operator()(get<array<next<T>>>(lhs_any));
    }
  }

  void operator()(array<void> &lhs) const {
    if (rhs != 0) {
      lhs_any = array<uint8_t>(lhs.size);
      operator()(get<array<uint8_t>>(lhs_any));
    }
  }

  void operator()(array<mp_int> &lhs) const {
    lhs[idx] += static_cast<mp_int>(rhs);
  }

  void operator()(array<weight_counter> &lhs) const { lhs[idx] += rhs; }
};

template <> struct radd_visitor<weight_counter> : public static_visitor<void> {
  any_array &lhs_any;
  const std::size_t idx;
  const weight_counter &rhs;
  radd_visitor(any_array &l, const std::size_t i, const weight_counter &r)
      : lhs_any(l), idx(i), rhs(r) {}

  template <typename T> void operator()(array<T> &lhs) const {
    lhs_any = array<weight_counter>(lhs);
    operator()(get<array<weight_counter>>(lhs_any));
  }

  void operator()(array<void> &lhs) const {
    lhs_any = array<weight_counter>(lhs.size);
    operator()(get<array<weight_counter>>(lhs_any));
  }

  void operator()(array<weight_counter> &lhs) const { lhs[idx] += rhs; }
};

// precondition: both arrays must have same size and may not be identical
struct radd_array_visitor : public static_visitor<void> {
  any_array &lhs_any;
  radd_array_visitor(any_array &l) : lhs_any(l) {}
  template <typename T> void operator()(const array<T> &rhs) const {
    for (auto i = 0ul; i < rhs.size; ++i)
      apply_visitor(radd_visitor<T>(lhs_any, i, rhs[i]), lhs_any);
  }
  void operator()(const array<void> &rhs) const {}
};

struct rmul_visitor : public static_visitor<void> {
  any_array &lhs_any;
  const double x;
  rmul_visitor(any_array &l, const double v) : lhs_any(l), x(v) {}
  template <typename T> void operator()(array<T> &lhs) const {
    lhs_any = array<weight_counter>(lhs);
    operator()(get<array<weight_counter>>(lhs_any));
  }
  void operator()(array<void> &) const {}
  void operator()(array<weight_counter> &lhs) const {
    for (auto i = 0ul; i != lhs.size; ++i)
      lhs[i] *= x;
  }
};

struct bicmp_visitor : public static_visitor<bool> {
  template <typename Array1, typename Array2>
  bool operator()(const Array1 &b1, const Array2 &b2) const {
    if (b1.size != b2.size)
      return false;
    return std::equal(b1.begin(), b1.end(), b2.begin());
  }

  template <typename T>
  bool operator()(const array<T> &b1, const array<void> &b2) const {
    if (b1.size != b2.size)
      return false;
    return std::all_of(b1.begin(), b1.end(), [](const T &t) { return t == 0; });
  }

  template <typename T>
  bool operator()(const array<void> &b1, const array<T> &b2) const {
    return operator()(b2, b1);
  }

  bool operator()(const array<void> &b1, const array<void> &b2) const {
    return b1.size == b2.size;
  }
};

} // namespace detail

class adaptive_storage {
  using buffer_type = detail::any_array;

public:
  using value_type = double;

  explicit adaptive_storage(std::size_t s) : buffer_(detail::array<void>(s)) {}

  adaptive_storage() = default;
  adaptive_storage(const adaptive_storage &) = default;
  adaptive_storage &operator=(const adaptive_storage &) = default;
  adaptive_storage(adaptive_storage &&) = default;
  adaptive_storage &operator=(adaptive_storage &&) = default;

  template <typename RHS, typename = detail::is_storage<RHS>>
  explicit adaptive_storage(const RHS &rhs)
      : buffer_(detail::array<void>(rhs.size())) {
    using T = typename RHS::value_type;
    for (auto i = 0ul, n = rhs.size(); i < n; ++i) {
      apply_visitor(detail::assign_visitor<T>(buffer_, i, rhs.value(i)),
                    buffer_);
    }
  }

  template <typename RHS> adaptive_storage &operator=(const RHS &rhs) {
    using T = typename RHS::value_type;
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      const auto n = rhs.size();
      if (size() != n) {
        buffer_ = detail::array<void>(n);
      }
      for (auto i = 0ul; i < n; ++i) {
        apply_visitor(detail::assign_visitor<T>(buffer_, i, rhs.value(i)),
                      buffer_);
      }
    }
    return *this;
  }

  std::size_t size() const {
    return apply_visitor(detail::size_visitor(), buffer_);
  }

  void increase(std::size_t i) {
    apply_visitor(detail::increase_visitor(buffer_, i), buffer_);
  }

  template <typename T> void add(std::size_t i, const T &value) {
    apply_visitor(detail::radd_visitor<T>(buffer_, i, value), buffer_);
  }

  template <typename T>
  void add(std::size_t i, const T &value, const T &variance) {
    if (value == variance) {
      apply_visitor(detail::radd_visitor<T>(buffer_, i, value), buffer_);
    } else {
      apply_visitor(detail::radd_visitor<detail::weight_counter>(
                        buffer_, i, detail::weight_counter(value, variance)),
                    buffer_);
    }
  }

  void weighted_increase(std::size_t i, const double weight_counter) {
    apply_visitor(detail::wincrease_visitor(buffer_, i, weight_counter), buffer_);
  }

  value_type value(std::size_t i) const {
    return apply_visitor(detail::value_visitor(i), buffer_);
  }

  value_type variance(std::size_t i) const {
    return apply_visitor(detail::variance_visitor(i), buffer_);
  }

  bool operator==(const adaptive_storage &rhs) const {
    return apply_visitor(detail::bicmp_visitor(), buffer_, rhs.buffer_);
  }

  // precondition: storages have same size
  adaptive_storage &operator+=(const adaptive_storage &rhs) {
    if (this == &rhs) {
      for (auto i = 0ul, n = size(); i < n; ++i)
        add(i, rhs.value(i), rhs.variance(i)); // this is losing precision
    } else {
      apply_visitor(detail::radd_array_visitor(buffer_), rhs.buffer_);
    }
    return *this;
  }

  // precondition: storages have same size
  template <typename RHS, typename = detail::is_storage<RHS>>
  adaptive_storage &operator+=(const RHS &rhs) {
    for (auto i = 0ul, n = size(); i < n; ++i)
      apply_visitor(detail::radd_visitor<typename RHS::value_type>(
                        buffer_, i, rhs.value(i)),
                    buffer_);
    return *this;
  }

  adaptive_storage &operator*=(const value_type x) {
    apply_visitor(detail::rmul_visitor(buffer_, x), buffer_);
    return *this;
  }

private:
  buffer_type buffer_;

  friend class ::boost::python::access;
  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

} // namespace histogram
} // namespace boost

#endif
