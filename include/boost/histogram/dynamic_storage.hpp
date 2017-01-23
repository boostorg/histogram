// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DYNAMIC_STORAGE_HPP_
#define _BOOST_HISTOGRAM_DYNAMIC_STORAGE_HPP_

#include <boost/histogram/detail/weight.hpp>
#include <boost/histogram/detail/mpl.hpp>
#include <boost/assert.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include <boost/cstdint.hpp>
#include <cstdint>
#include <cstddef>
#include <cstdlib> // malloc/free
#include <new> // for bad_alloc exception
#include <type_traits>
#include <algorithm>
#include <limits>

namespace boost {
namespace histogram {

namespace detail {

  static_assert(std::is_pod<weight_t>::value, "weight_t must be POD");

  using type_to_int = mpl::map<
    mpl::pair<weight_t, mpl::int_<-1>>,

    mpl::pair<int8_t, mpl::int_<1>>,
    mpl::pair<int16_t, mpl::int_<2>>,
    mpl::pair<int32_t, mpl::int_<3>>,
    mpl::pair<int64_t, mpl::int_<4>>,

    mpl::pair<uint8_t, mpl::int_<1>>,
    mpl::pair<uint16_t, mpl::int_<2>>,
    mpl::pair<uint32_t, mpl::int_<3>>,
    mpl::pair<uint64_t, mpl::int_<4>>
  >;

  using int_to_type = mpl::map<
    mpl::pair<mpl::int_<-1>, weight_t>,
    mpl::pair<mpl::int_<1>, uint8_t>,
    mpl::pair<mpl::int_<2>, uint16_t>,
    mpl::pair<mpl::int_<3>, uint32_t>,
    mpl::pair<mpl::int_<4>, uint64_t>
  >;

  template <typename T>
  using storage_type =
    typename mpl::at<
      detail::int_to_type,
      typename mpl::at<
        detail::type_to_int, T
      >::type
    >::type;

  template <typename T>
  using next_storage_type =
    typename mpl::at<int_to_type,
      typename mpl::next<
        typename mpl::at<type_to_int, T>::type
      >::type
    >::type;

  struct buffer
  {
    explicit buffer(std::size_t s = 0) :
      size_(s),
      ptr_(nullptr)
    {}

    buffer(const buffer& o) :
      type_(o.type_), size_(o.size_), ptr_(nullptr)
    {
      realloc(type_.depth_);
      std::copy(static_cast<const char*>(o.ptr_),
                static_cast<const char*>(o.ptr_) + size_ * type_.depth_,
                static_cast<char*>(ptr_));
    }

    buffer& operator=(const buffer& o)
    {
      if (this != &o) {
        if (size_ != o.size_ || type_.id_ != o.type_.id_) {
          size_ = o.size_;
          type_ = o.type_;
          realloc(type_.depth_);
        }
        std::copy(static_cast<const char*>(o.ptr_),
                  static_cast<const char*>(o.ptr_) + size_ * type_.depth_,
                  static_cast<char*>(ptr_));
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
      type_.set<T>();
      ptr_ = std::malloc(size_ * sizeof(T));
      new (ptr_) T[size_];
    }

    template <typename T>
    void destroy() {
      std::free(ptr_);
      ptr_ = nullptr;
    }

    void destroy_any() {
      switch (type_.id_) {
        case -1: destroy<weight_t>(); break;
        case 0: /* do nothing */ break;
        case 1: destroy<uint8_t>(); break;
        case 2: destroy<uint16_t>(); break;
        case 3: destroy<uint32_t>(); break;
        case 4: destroy<uint64_t>(); break;
      }
    }

    template <typename T,
              typename U = next_storage_type<T>>
    void grow() {
      static_assert(sizeof(U) >= sizeof(T), "U must be as large or larger than T");
      realloc(sizeof(U));
      std::copy_backward(&at<T>(0), &at<T>(size_), &at<U>(size_));
      type_.set<U>();
    }

    void wconvert()
    {
      switch (type_.id_) {
        case -1: /* do nothing */ break;
        case 0: initialize<weight_t>(); break;
        case 1: grow<uint8_t, weight_t> (); break;
        case 2: grow<uint16_t, weight_t>(); break;
        case 3: grow<uint32_t, weight_t>(); break;
        case 4: grow<uint64_t, weight_t>(); break;
      }
    }

    template <typename T>
    void initialize() {
      type_.set<T>();
      ptr_ = nullptr;
      realloc(sizeof(T));
      std::fill(&at<T>(0), &at<T>(size_), T(0));
    }

    void realloc(unsigned d)
    {
      ptr_ = std::realloc(ptr_, size_ * d);
      if (!ptr_ && (size_ * d > 0))
        throw std::bad_alloc();
    }

    template <typename T>
    T& at(std::size_t i) { return static_cast<T*>(ptr_)[i]; }

    template <typename T>
    const T& at(std::size_t i) const { return static_cast<const T*>(ptr_)[i]; }

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

  template <typename T>
  void increase_impl(buffer& b, std::size_t i) {
    auto& bi = b.at<T>(i);
    if (bi < std::numeric_limits<T>::max())
      ++bi;
    else {
      b.grow<T>();
      using U = detail::next_storage_type<T>;
      ++b.at<U>(i);
    }
  }

  template <>
  void increase_impl<uint64_t>(buffer& b, std::size_t i) {
    auto& bi = b.at<uint64_t>(i);
    if (bi < std::numeric_limits<uint64_t>::max())
      ++bi;
    else
      throw std::overflow_error("histogram overflow");
  }

  template <typename T, typename TO>
  struct add_one_impl {
    static void apply(buffer& b, std::size_t i, const TO& o) {
      auto& bi = b.at<T>(i);
      if (static_cast<T>(std::numeric_limits<T>::max() - bi) >= o)
        bi += o;
      else {
        b.grow<T>();
        add_one_impl<detail::next_storage_type<T>, TO>::apply(b, i, o);
      }
    }
  };

  template <typename TO>
  struct add_one_impl<uint64_t, TO> {
    static void apply(buffer& b, std::size_t i, const TO& o) {
      auto& bi = b.at<uint64_t>(i);
      if (static_cast<uint64_t>(std::numeric_limits<uint64_t>::max() - bi) >= o)
        bi += o;
      else
        throw std::overflow_error("histogram overflow");
    }
  };

  template <typename TO>
  void add_impl(buffer& b, const buffer& o) {
    for (decltype(b.size_) i = 0; i < b.size_; ++i)
      switch (b.type_.id_) {
        case -1: b.at<weight_t>(i) += o.at<TO>(i); break;
        case 0: b.initialize<uint8_t>(); // fall through
        case 1: add_one_impl<uint8_t, TO>::apply(b, i, o.at<TO>(i)); break;
        case 2: add_one_impl<uint16_t, TO>::apply(b, i, o.at<TO>(i)); break;
        case 3: add_one_impl<uint32_t, TO>::apply(b, i, o.at<TO>(i)); break;
        case 4: add_one_impl<uint64_t, TO>::apply(b, i, o.at<TO>(i)); break;
      }
  }

  template <>
  void add_impl<weight_t>(buffer& b, const buffer& o) {
    b.wconvert();
    for (decltype(b.size_) i = 0; i < b.size_; ++i)
      b.at<weight_t>(i) += o.at<weight_t>(i);
  }

} // NS detail

class dynamic_storage
{
public:
  using value_t = double;
  using variance_t = double;

  explicit
  dynamic_storage(std::size_t s) :
    buffer_(s)
  {}

  dynamic_storage() = default;
  dynamic_storage(const dynamic_storage&) = default;
  dynamic_storage& operator=(const dynamic_storage&) = default;
  dynamic_storage(dynamic_storage&&) = default;
  dynamic_storage& operator=(dynamic_storage&&) = default;

  template <typename T,
            template <typename> class Storage,
            typename = detail::is_standard_integral<T>>
  dynamic_storage(const Storage<T>& o) :
    buffer_(o.size())
  {
    using U = detail::storage_type<T>;
    buffer_.create<U>();
    std::copy(o.data(), o.data() + buffer_.size_, &buffer_.at<U>(0));
  }

  template <typename T,
            template <typename> class Storage,
            typename = detail::is_standard_integral<T>>
  dynamic_storage& operator=(const Storage<T>& o)
  {
    buffer_.destroy_any();
    buffer_.size_ = o.size();
    using U = detail::storage_type<T>;
    buffer_.create<U>();
    std::copy(o.data(), o.data() + buffer_.size_, &buffer_.at<U>(0));
    return *this;
  }

  std::size_t size() const { return buffer_.size_; }
  unsigned depth() const { return buffer_.type_.depth_; }
  const void* data() const { return buffer_.ptr_; }
  void increase(std::size_t i);
  void increase(std::size_t i, double w);
  value_t value(std::size_t i) const;
  variance_t variance(std::size_t i) const;
  dynamic_storage& operator+=(const dynamic_storage&);

private:
  detail::buffer buffer_;

  template <class Archive>
  friend void serialize(Archive&, dynamic_storage&, unsigned);
};

inline
void dynamic_storage::increase(std::size_t i)
{
  switch (buffer_.type_.id_) {
    case -1: ++(buffer_.at<detail::weight_t>(i)); break;
    case 0: buffer_.initialize<uint8_t>(); // and fall through
    case 1: detail::increase_impl<uint8_t> (buffer_, i); break;
    case 2: detail::increase_impl<uint16_t>(buffer_, i); break;
    case 3: detail::increase_impl<uint32_t>(buffer_, i); break;
    case 4: detail::increase_impl<uint64_t>(buffer_, i); break;
  }
}

inline
void dynamic_storage::increase(std::size_t i, double w)
{
  buffer_.wconvert();
  buffer_.at<detail::weight_t>(i).add_weight(w);
}

inline
dynamic_storage& dynamic_storage::operator+=(const dynamic_storage& o)
{
  switch (o.buffer_.type_.id_) {
    case -1: detail::add_impl<detail::weight_t>(buffer_, o.buffer_); break;
    case 0: /* do nothing */ break;
    case 1: detail::add_impl<uint8_t>(buffer_, o.buffer_); break;
    case 2: detail::add_impl<uint16_t>(buffer_, o.buffer_); break;
    case 3: detail::add_impl<uint32_t>(buffer_, o.buffer_); break;
    case 4: detail::add_impl<uint64_t>(buffer_, o.buffer_); break;
  }
  return *this;
}

inline
dynamic_storage::value_t dynamic_storage::value(std::size_t i) const
{
  switch (buffer_.type_.id_) {
    case -1: return buffer_.at<detail::weight_t>(i).w;
    case 0: /* do nothing */ break;
    case 1: return buffer_.at<uint8_t> (i);
    case 2: return buffer_.at<uint16_t>(i);
    case 3: return buffer_.at<uint32_t>(i);
    case 4: return buffer_.at<uint64_t>(i);
  }
  return 0.0;
}

inline
dynamic_storage::variance_t dynamic_storage::variance(std::size_t i) const
{
  switch (buffer_.type_.id_) {
    case -1: return buffer_.at<detail::weight_t>(i).w2;
    case 0: /* do nothing */ break;
    case 1: return buffer_.at<uint8_t> (i);
    case 2: return buffer_.at<uint16_t>(i);
    case 3: return buffer_.at<uint32_t>(i);
    case 4: return buffer_.at<uint64_t>(i);
  }
  return 0.0;
}

}
}

#endif
