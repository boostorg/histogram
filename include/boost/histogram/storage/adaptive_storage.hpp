// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP_
#define _BOOST_HISTOGRAM_STORAGE_ADAPTIVE_HPP_

#include <boost/histogram/detail/weight.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/assert.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include <boost/cstdint.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <type_traits>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <memory>

// forward declaration for serialization workaround
namespace boost { namespace serialization { class access; }}

namespace boost {
namespace histogram {

namespace detail {

  using mp_int = multiprecision::cpp_int;

  using type_to_int = mpl::map<
    mpl::pair<weight, mpl::int_<-1>>,

    mpl::pair<int8_t, mpl::int_<1>>,
    mpl::pair<int16_t, mpl::int_<2>>,
    mpl::pair<int32_t, mpl::int_<3>>,
    mpl::pair<int64_t, mpl::int_<4>>,

    mpl::pair<uint8_t, mpl::int_<1>>,
    mpl::pair<uint16_t, mpl::int_<2>>,
    mpl::pair<uint32_t, mpl::int_<3>>,
    mpl::pair<uint64_t, mpl::int_<4>>,

    mpl::pair<mp_int, mpl::int_<5>>
  >;

  using int_to_type = mpl::map<
    mpl::pair<mpl::int_<-1>, weight>,
    mpl::pair<mpl::int_<1>, uint8_t>,
    mpl::pair<mpl::int_<2>, uint16_t>,
    mpl::pair<mpl::int_<3>, uint32_t>,
    mpl::pair<mpl::int_<4>, uint64_t>,
    mpl::pair<mpl::int_<5>, mp_int>
  >;

  template <typename T>
  using storage_type =
    typename mpl::at<
      detail::int_to_type,
      typename mpl::at<detail::type_to_int, T>::type
    >::type;

  template <typename T>
  using next_storage_type =
    typename mpl::at<int_to_type,
      typename mpl::next<
        typename mpl::at<type_to_int, T>::type
      >::type
    >::type;

  template <typename Buffer, typename T, typename TO>
  struct add_one_impl {
    static void apply(Buffer& b, const std::size_t i, const TO& o) {
      auto& bi = b.template at<T>(i);
      if (static_cast<T>(std::numeric_limits<T>::max() - bi) >= o)
        bi += static_cast<T>(o);
      else {
        b.template grow<T>();
        add_one_impl<Buffer, next_storage_type<T>, TO>::apply(b, i, o);
      }
    }
  };

  template <typename Buffer, typename TO>
  struct add_one_impl<Buffer, mp_int, TO> {
    static void apply(Buffer& b, const std::size_t i, const TO& o) {
      b.template at<mp_int>(i) += o;
    }
  };

  template <typename Buffer, typename OtherBuffer, typename TO>
  struct add_impl {
    static void apply(Buffer& b, const OtherBuffer& o) {
      for (std::size_t i = 0; i < b.size_; ++i) {
        const auto oi = o.template at<TO>(i);
        switch (b.type_.id_) {
          case -1: b.template at<weight>(i) += oi; break;
          case 0: b.template initialize<uint8_t>(); // fall through
          case 1: add_one_impl<Buffer, uint8_t, TO>::apply(b, i, oi); break;
          case 2: add_one_impl<Buffer, uint16_t, TO>::apply(b, i, oi); break;
          case 3: add_one_impl<Buffer, uint32_t, TO>::apply(b, i, oi); break;
          case 4: add_one_impl<Buffer, uint64_t, TO>::apply(b, i, oi); break;
          case 5: add_one_impl<Buffer, mp_int, TO>::apply(b, i, oi); break;
        }
      }
    }
  };

  template <typename Buffer, typename OtherBuffer>
  struct add_impl<Buffer, OtherBuffer, weight> {
    static void apply(Buffer& b, const OtherBuffer& o) {
      b.wconvert();
      for (std::size_t i = 0; i < b.size_; ++i)
        b.template at<weight>(i) += o.template at<weight>(i);
    }
  };

  template <typename Buffer, typename OtherBuffer, typename  T>
  struct cmp_impl {
    static bool apply(const Buffer& b, const OtherBuffer& o) {
      switch (o.type_.id_) {
        case -1:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (b.template at<T>(i) != o.template at<weight>(i))
              return false;
          break;
        case 0:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (b.template at<T>(i) != 0)
              return false;
          break;
        case 1:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (b.template at<T>(i) != o.template at<uint8_t>(i))
              return false;
          break;
        case 2:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (b.template at<T>(i) != o.template at<uint16_t>(i))
              return false;
          break;
        case 3:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (b.template at<T>(i) != o.template at<uint32_t>(i))
              return false;
          break;
        case 4:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (b.template at<T>(i) != o.template at<uint64_t>(i))
              return false;
          break;
        case 5:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (b.template at<T>(i) != o.template at<mp_int>(i))
              return false;
          break;
      }
      return true;
    }
  };

  template <typename Buffer, typename OtherBuffer>
  struct cmp_impl<Buffer, OtherBuffer, void> {
    static bool apply(const Buffer& b, const OtherBuffer& o) {
      switch (o.type_.id_) {
        case -1:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (0 != o.template at<weight>(i))
              return false;
          break;
        case 0: break;
        case 1:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (0 != o.template at<uint8_t>(i))
              return false;
          break;
        case 2:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (0 != o.template at<uint16_t>(i))
              return false;
          break;
        case 3:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (0 != o.template at<uint32_t>(i))
              return false;
          break;
        case 4:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (0 != o.template at<uint64_t>(i))
              return false;
          break;
        case 5:
          for (std::size_t i = 0; i < b.size_; ++i)
            if (0 != o.template at<mp_int>(i))
              return false;
          break;
      }
      return true;
    }
  };

  template <template<class> class Allocator>
  struct buffer
  {
    explicit buffer(std::size_t s = 0) :
      size_(s), ptr_(nullptr)
    {}

    buffer(const buffer& o) :
      size_(o.size_)
    {
      switch (o.type_.id_) {
        case -1: create<weight>(); copy_from<weight>(o.ptr_); break;
        case 0: type_ = type(); ptr_ = nullptr; break;
        case 1: create<uint8_t>(); copy_from<uint8_t>(o.ptr_); break;
        case 2: create<uint16_t>(); copy_from<uint16_t>(o.ptr_); break;
        case 3: create<uint32_t>(); copy_from<uint32_t>(o.ptr_); break;
        case 4: create<uint64_t>(); copy_from<uint64_t>(o.ptr_); break;
        case 5: create<mp_int>(); copy_from<mp_int>(o.ptr_); break;
      }
    }

    buffer& operator=(const buffer& o)
    {
      if (this != &o) {
        if (size_ != o.size_ || type_.id_ != o.type_.id_) {
          destroy_any();
          size_ = o.size_;
          switch (o.type_.id_) {
            case -1: create<weight>(); copy_from<weight>(o.ptr_); break;
            case 0: type_ = type(); ptr_ = nullptr; break;
            case 1: create<uint8_t>(); copy_from<uint8_t>(o.ptr_); break;
            case 2: create<uint16_t>(); copy_from<uint16_t>(o.ptr_); break;
            case 3: create<uint32_t>(); copy_from<uint32_t>(o.ptr_); break;
            case 4: create<uint64_t>(); copy_from<uint64_t>(o.ptr_); break;
            case 5: create<mp_int>(); copy_from<mp_int>(o.ptr_); break;
          }
        } else {
          switch (o.type_.id_) {
            case -1: copy_from<weight>(o.ptr_); break;
            case 0: type_ = type(); ptr_ = nullptr; break;
            case 1: copy_from<uint8_t>(o.ptr_); break;
            case 2: copy_from<uint16_t>(o.ptr_); break;
            case 3: copy_from<uint32_t>(o.ptr_); break;
            case 4: copy_from<uint64_t>(o.ptr_); break;
            case 5: copy_from<mp_int>(o.ptr_); break;
          }
        }
      }
      return *this;
    }

    buffer(buffer&& o) :
      type_(o.type_),
      size_(o.size_),
      ptr_(o.ptr_)
    {
      o.size_ = 0;
      o.type_ = type();
      o.ptr_ = nullptr;
    }

    buffer& operator=(buffer&& o)
    {
      if (this != &o) {
        destroy_any();
        type_ = o.type_;
        size_ = o.size_;
        ptr_ = o.ptr_;
        o.type_ = type();
        o.size_ = 0;
        o.ptr_ = nullptr;
      }
      return *this;
    }

    ~buffer() { destroy_any(); }

    template <typename T>
    void create() {
      Allocator<T> a;
      ptr_ = a.allocate(size_);
      new (ptr_) T[size_];
      type_.template set<T>();
    }

    template <typename T>
    void destroy() {
      for (T* iter = &at<T>(0); iter != &at<T>(size_); ++iter)
        iter->~T();
      Allocator<T> a;
      a.deallocate(static_cast<T*>(ptr_), size_);
      ptr_ = nullptr;
    }

    void destroy_any() {
      switch (type_.id_) {
        case -1: destroy<weight>(); break;
        case 0: ptr_ = nullptr; break;
        case 1: destroy<uint8_t>(); break;
        case 2: destroy<uint16_t>(); break;
        case 3: destroy<uint32_t>(); break;
        case 4: destroy<uint64_t>(); break;
        case 5: destroy<mp_int>(); break;
      }
    }

    template <typename T>
    void copy_from(const void* p) {
      std::copy(static_cast<const T*>(p), static_cast<const T*>(p) + size_,
                static_cast<T*>(ptr_));
    }

    template <typename T,
              typename U = next_storage_type<T>>
    void grow() {
      Allocator<U> a;
      U* u = a.allocate(size_);
      new (u) U[size_];
      std::copy(&at<T>(0), &at<T>(size_), u);
      destroy<T>();
      ptr_ = u;
      type_.template set<U>();
    }

    void wconvert()
    {
      switch (type_.id_) {
        case -1: break;
        case 0: initialize<weight>(); break;
        case 1: grow<uint8_t, weight> (); break;
        case 2: grow<uint16_t, weight>(); break;
        case 3: grow<uint32_t, weight>(); break;
        case 4: grow<uint64_t, weight>(); break;
        case 5: grow<mp_int, weight>(); break;
      }
    }

    template <typename T>
    void initialize() {
      create<T>();
      std::fill(&at<T>(0), &at<T>(size_), T(0));
    }

    template <typename T>
    inline
    T& at(std::size_t i) { return static_cast<T*>(ptr_)[i]; }

    template <typename T>
    inline
    const T& at(std::size_t i) const { return static_cast<const T*>(ptr_)[i]; }

    template <typename T>
    inline
    void increase(const std::size_t i) {
      auto& bi = at<T>(i);
      if (bi < std::numeric_limits<T>::max())
        ++bi;
      else {
        grow<T>();
        ++at<next_storage_type<T>>(i);
      }
    }

    struct type {
      char id_ = 0, depth_ = 0;
      template <typename T>
      void set() {
        id_ = mpl::at<type_to_int, T>::type::value;
        depth_ = sizeof(T);
      }
    } type_;
    std::size_t size_;
    void* ptr_;
  };

} // NS detail

template <template<class> class Allocator=std::allocator>
class adaptive_storage
{
public:
  using value_type = double;

  explicit
  adaptive_storage(std::size_t s) :
    buffer_(s)
  {}

  adaptive_storage() = default;
  adaptive_storage(const adaptive_storage&) = default;
  adaptive_storage& operator=(const adaptive_storage&) = default;
  adaptive_storage(adaptive_storage&&) = default;
  adaptive_storage& operator=(adaptive_storage&&) = default;

  template <typename OtherStorage,
            typename = detail::is_standard_integral<typename OtherStorage::value_type>>
  adaptive_storage(const OtherStorage& o);

  template <typename OtherStorage,
            typename = detail::is_standard_integral<typename OtherStorage::value_type>>
  adaptive_storage& operator=(const OtherStorage& o);

  std::size_t size() const { return buffer_.size_; }
  void increase(std::size_t i);
  void increase(std::size_t i, double w);
  value_type value(std::size_t i) const;
  value_type variance(std::size_t i) const;

  template <template <class> class Allocator1>
  adaptive_storage& operator+=(const adaptive_storage<Allocator1>&);

  template <typename OtherStorage,
            typename = detail::is_storage<OtherStorage>>
  adaptive_storage& operator+=(const OtherStorage& other);

  template <template <class> class Allocator1>
  bool operator==(const adaptive_storage<Allocator1>&) const;

  template <typename OtherStorage,
            typename = detail::is_not_adaptive_storage<OtherStorage>>
  bool operator==(const OtherStorage& other) const;

private:
  detail::buffer<Allocator> buffer_;

  // workaround for gcc-4.8
  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);

  friend struct storage_access;
};

template <template <class> class Allocator>
template <typename OtherStorage, typename>
adaptive_storage<Allocator>::adaptive_storage(const OtherStorage& o) :
  buffer_(o.size())
{
  using T = typename OtherStorage::value_type;
  using U = detail::storage_type<T>;
  buffer_.template create<U>();
  for (std::size_t i = 0; i < buffer_.size_; ++i)
    buffer_.template at<U>(i) = o.value(i);
}

template <template <class> class Allocator>
template <typename OtherStorage, typename>
adaptive_storage<Allocator>&
adaptive_storage<Allocator>::operator=(const OtherStorage& o)
{
  using T = typename OtherStorage::value_type;
  buffer_.destroy_any();
  buffer_.size_ = o.size();
  using U = detail::storage_type<T>;
  buffer_.template create<U>();
  for (std::size_t i = 0; i < buffer_.size_; ++i)
    buffer_.template at<U>(i) = o.value(i);
  return *this;
}

template <template <class> class Allocator>
inline
void
adaptive_storage<Allocator>::increase(std::size_t i)
{
  switch (buffer_.type_.id_) {
    case -1: ++(buffer_.template at<detail::weight>(i)); break;
    case 0: buffer_.template initialize<uint8_t>(); // and fall through
    case 1: buffer_.template increase<uint8_t>(i); break;
    case 2: buffer_.template increase<uint16_t>(i); break;
    case 3: buffer_.template increase<uint32_t>(i); break;
    case 4: buffer_.template increase<uint64_t>(i); break;
    case 5: ++(buffer_.template at<detail::mp_int>(i)); break;
  }
}

template <template <class> class Allocator>
inline
void
adaptive_storage<Allocator>::increase(std::size_t i, double w)
{
  buffer_.wconvert();
  buffer_.template at<detail::weight>(i).add_weight(w);
}

template <template <class> class Allocator>
inline
typename adaptive_storage<Allocator>::value_type
adaptive_storage<Allocator>::value(std::size_t i) const
{
  switch (buffer_.type_.id_) {
    case -1: return buffer_.template at<detail::weight>(i).w;
    case 0: break;
    case 1: return buffer_.template at<uint8_t> (i);
    case 2: return buffer_.template at<uint16_t>(i);
    case 3: return buffer_.template at<uint32_t>(i);
    case 4: return buffer_.template at<uint64_t>(i);
    case 5: return static_cast<double>(buffer_.template at<detail::mp_int>(i));
  }
  return 0.0;
}

template <template <class> class Allocator>
inline
typename adaptive_storage<Allocator>::value_type
adaptive_storage<Allocator>::variance(std::size_t i) const
{
  switch (buffer_.type_.id_) {
    case -1: return buffer_.template at<detail::weight>(i).w2;
    case 0: break;
    case 1: return buffer_.template at<uint8_t> (i);
    case 2: return buffer_.template at<uint16_t>(i);
    case 3: return buffer_.template at<uint32_t>(i);
    case 4: return buffer_.template at<uint64_t>(i);
    case 5: return static_cast<double>(buffer_.template at<detail::mp_int>(i));
  }
  return 0.0;
}

template <template <class> class Allocator>
template <template <class> class OtherAllocator>
adaptive_storage<Allocator>&
adaptive_storage<Allocator>::operator+=(const adaptive_storage<OtherAllocator>& o)
{
  using B = decltype(buffer_);
  using OB = decltype(o.buffer_);
  switch (o.buffer_.type_.id_) {
    case -1: detail::add_impl<B, OB, detail::weight>::apply(buffer_, o.buffer_); break;
    case 0: break;
    case 1: detail::add_impl<B, OB, uint8_t>::apply(buffer_, o.buffer_); break;
    case 2: detail::add_impl<B, OB, uint16_t>::apply(buffer_, o.buffer_); break;
    case 3: detail::add_impl<B, OB, uint32_t>::apply(buffer_, o.buffer_); break;
    case 4: detail::add_impl<B, OB, uint64_t>::apply(buffer_, o.buffer_); break;
    case 5: detail::add_impl<B, OB, detail::mp_int>::apply(buffer_, o.buffer_); break;
  }
  return *this;
}

template <template <class> class Allocator>
template <typename OtherStorage, typename>
adaptive_storage<Allocator>&
adaptive_storage<Allocator>::operator+=(const OtherStorage& other)
{
  using B = decltype(buffer_);
  using TO = typename OtherStorage::value_type;
  for (std::size_t i = 0; i < buffer_.size_; ++i) {
    const auto oi = other.value(i);
    switch (buffer_.type_.id_) {
      case -1: buffer_.template at<detail::weight>(i) += oi; break;
      case 0: buffer_.template initialize<uint8_t>(); // fall through
      case 1: detail::add_one_impl<B, uint8_t, TO>::apply(buffer_, i, oi); break;
      case 2: detail::add_one_impl<B, uint16_t, TO>::apply(buffer_, i, oi); break;
      case 3: detail::add_one_impl<B, uint32_t, TO>::apply(buffer_, i, oi); break;
      case 4: detail::add_one_impl<B, uint64_t, TO>::apply(buffer_, i, oi); break;
      case 5: detail::add_one_impl<B, detail::mp_int, TO>::apply(buffer_, i, oi); break;
    }
  }
  return *this;
}

template <template <class> class Allocator1>
template <template <class> class Allocator2>
bool
adaptive_storage<Allocator1>::operator==(const adaptive_storage<Allocator2>& o) const
{
  if (buffer_.size_ != o.buffer_.size_)
    return false;
  using TB = decltype(buffer_);
  using OB = decltype(o.buffer_);
  switch (buffer_.type_.id_) {
    case -1: return detail::cmp_impl<TB, OB, detail::weight>::apply(buffer_, o.buffer_);
    case 0: return detail::cmp_impl<TB, OB, void>::apply(buffer_, o.buffer_);
    case 1: return detail::cmp_impl<TB, OB, uint8_t>::apply(buffer_, o.buffer_);
    case 2: return detail::cmp_impl<TB, OB, uint16_t>::apply(buffer_, o.buffer_);
    case 3: return detail::cmp_impl<TB, OB, uint32_t>::apply(buffer_, o.buffer_);
    case 4: return detail::cmp_impl<TB, OB, uint64_t>::apply(buffer_, o.buffer_);
    case 5: return detail::cmp_impl<TB, OB, detail::mp_int>::apply(buffer_, o.buffer_);
  }
  return false;
}

template <template <class> class Allocator>
template <typename OtherStorage, typename>
bool
adaptive_storage<Allocator>::operator==(const OtherStorage& o) const
{
  if (buffer_.size_ != o.size())
    return false;
  switch (buffer_.type_.id_) {
    case -1:
      for (std::size_t i = 0; i < buffer_.size_; ++i)
        if (buffer_.template at<detail::weight>(i).w != o.value(i) ||
            buffer_.template at<detail::weight>(i).w2 != o.variance(i))
          return false;
      break;
    case 0:
      for (std::size_t i = 0; i < buffer_.size_; ++i)
        if (0 != o.value(i) || 0 != o.variance(i))
          return false;
      break;
    case 1:
      for (std::size_t i = 0; i < buffer_.size_; ++i)
        if (buffer_.template at<uint8_t>(i) != o.value(i) ||
            buffer_.template at<uint8_t>(i) != o.variance(i))
          return false;
      break;
    case 2:
      for (std::size_t i = 0; i < buffer_.size_; ++i)
        if (buffer_.template at<uint16_t>(i) != o.value(i) ||
            buffer_.template at<uint16_t>(i) != o.variance(i))
          return false;
      break;
    case 3:
      for (std::size_t i = 0; i < buffer_.size_; ++i)
        if (buffer_.template at<uint32_t>(i) != o.value(i) ||
            buffer_.template at<uint32_t>(i) != o.variance(i))
          return false;
      break;
    case 4:
      for (std::size_t i = 0; i < buffer_.size_; ++i)
        if (buffer_.template at<uint64_t>(i) != o.value(i) ||
            buffer_.template at<uint64_t>(i) != o.variance(i))
          return false;
      break;
    case 5:
      for (std::size_t i = 0; i < buffer_.size_; ++i)
        if (buffer_.template at<detail::mp_int>(i) != o.value(i) ||
            buffer_.template at<detail::mp_int>(i) != o.variance(i))
          return false;
      break;
  }
  return true;
}

template <template <class> class Allocator,
          typename OtherStorage,
          typename = detail::is_not_adaptive_storage<OtherStorage>>
inline
bool
operator==(const OtherStorage& a, const adaptive_storage<Allocator>& b)
{
  return b == a;
}

} // NS histogram
} // NS boost

#endif
