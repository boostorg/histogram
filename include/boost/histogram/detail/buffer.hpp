// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_BUFFER_HPP_
#define _BOOST_HISTOGRAM_DETAIL_BUFFER_HPP_

#include <cstdlib>
#include <algorithm>
#include <new> // for bad_alloc exception
#include <boost/mpl/map.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include <boost/histogram/detail/weight.hpp>

namespace boost {
namespace histogram {
namespace detail {

  using type_to_int = mpl::map<
    mpl::pair<int8_t, mpl::int_<1>>,
    mpl::pair<int16_t, mpl::int_<2>>,
    mpl::pair<int32_t, mpl::int_<3>>,
    mpl::pair<int64_t, mpl::int_<4>>,
    mpl::pair<uint8_t, mpl::int_<1>>,
    mpl::pair<uint16_t, mpl::int_<2>>,
    mpl::pair<uint32_t, mpl::int_<3>>,
    mpl::pair<uint64_t, mpl::int_<4>>,
    mpl::pair<weight_t, mpl::int_<6>>
  >;

  using int_to_type = mpl::map<
    mpl::pair<mpl::int_<1>, uint8_t>,
    mpl::pair<mpl::int_<2>, uint16_t>,
    mpl::pair<mpl::int_<3>, uint32_t>,
    mpl::pair<mpl::int_<4>, uint64_t>,
    mpl::pair<mpl::int_<6>, weight_t>
  >;

  template <typename T>
  using next_storage_type =
    typename mpl::at<int_to_type,
      typename mpl::next<
        typename mpl::at<type_to_int, T>::type
      >::type
    >::type;

  class buffer {
  public:

    explicit
    buffer(std::size_t s = 0) :
      size_(s),
      type_(0),
      ptr_(nullptr)
    {}

    buffer(const buffer& o) :
      size_(o.size_), type_(o.type_), ptr_(nullptr)
    {
      realloc(depth());
      std::copy(static_cast<const char*>(o.ptr_),
                static_cast<const char*>(o.ptr_) + size_ * depth(),
                static_cast<char*>(ptr_));
    }

    buffer& operator=(const buffer& o)
    {
      if (this != &o) {
        if (size_ != o.size_ || type_ != o.type_) {
          size_ = o.size_;
          type_ = o.type_;
          realloc(depth());
        }
        std::copy(static_cast<const char*>(o.ptr_),
                  static_cast<const char*>(o.ptr_) + size_ * depth(),
                  static_cast<char*>(ptr_));
      }
      return *this;
    }

    template <typename T, template<typename> class Storage>
    buffer(const Storage<T>& o) :
      size_(o.size()),
      type_(mpl::at<type_to_int, T>::type::value),
      ptr_(nullptr)
    {
      using U = typename mpl::at<
          int_to_type,
          typename mpl::at<type_to_int, T>::type
        >::type;
      realloc(sizeof(U));
      std::copy(o.data_, o.data_ + size_, &at<U>(0));
    }

    template <typename T, template<typename> class Storage>
    buffer& operator=(const Storage<T>& o) {
      size_ = o.size();
      type_ = mpl::at<type_to_int, T>::type::value;
      using U = typename mpl::at<
          int_to_type,
          typename mpl::at<type_to_int, T>::type
        >::type;
      realloc(sizeof(U));
      std::copy(o.data_, o.data_ + size_, &at<T>(0));
      return *this;
    }

    buffer(buffer&& o) :
      size_(o.size_),
      type_(o.type_),
      ptr_(o.ptr_)
    {
      o.size_ = 0;
      o.type_ = 0;
      o.ptr_ = nullptr;
    }

    buffer& operator=(buffer&& o)
    {
      if (this != &o) {
        std::free(ptr_);
        size_ = o.size_;
        type_ = o.type_;
        ptr_ = o.ptr_;
        o.size_ = 0;
        o.type_ = 0;
        o.ptr_ = nullptr;
      }
      return *this;
    }

    template <typename T, template<typename> class Storage>
    buffer(Storage<T>&& o) :
      size_(o.size_),
      type_(mpl::at<type_to_int, T>::type::value),
      ptr_(const_cast<void*>(o.data_))
    {
      o.size_ = 0;
      o.data_ = nullptr;
    }

    template <typename T, template<typename> class Storage>
    buffer& operator=(Storage<T>&& o)
    {
      std::free(ptr_);
      size_ = o.size_;
      type_ = mpl::at<type_to_int, T>::type::value;
      ptr_ = static_cast<void*>(o.data_);
      o.size_ = 0;
      o.data_ = nullptr;
      return *this;
    }

    ~buffer() { std::free(ptr_); }

    std::size_t size() const { return size_; }

    unsigned type() const { return type_; }

    unsigned depth() const {
      switch (type_) {
        case 1: return sizeof(uint8_t);
        case 2: return sizeof(uint16_t);
        case 3: return sizeof(uint32_t);
        case 4: return sizeof(uint64_t);
        case 6: return sizeof(weight_t);
      }
      return 0;
    }

    const void* data() const { return ptr_; }

    bool operator==(const buffer& o) const {
      return size_ == o.size_ &&
             type_ == o.type_ &&
             std::equal(static_cast<char*>(ptr_),
                        static_cast<char*>(ptr_) + size_ * depth(),
                        static_cast<char*>(o.ptr_));
    }

    template <typename T,
              typename U = next_storage_type<T>>
    void grow() {
      static_assert(sizeof(U) >= sizeof(T), "U must be as large or larger than T");
      type_ = mpl::at<type_to_int, U>::type::value;
      realloc(sizeof(U));
      std::copy_backward(&at<T>(0), &at<T>(size_), &at<U>(size_));
    }

    template <typename T>
    void initialize() {
      type_ = mpl::at<type_to_int, T>::type::value;
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

  private:
    std::size_t size_;
    unsigned type_;
    void* ptr_;

    template <class Archive>
    friend void serialize(Archive&, buffer&, unsigned);
  };

}
}
}

#endif
