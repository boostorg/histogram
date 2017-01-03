// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STATIC_STORAGE_HPP_
#define _BOOST_HISTOGRAM_STATIC_STORAGE_HPP_

#include <boost/histogram/detail/buffer.hpp>

namespace boost {
namespace histogram {

  template <typename T>
  class static_storage {
    using buffer_t = detail::buffer_t;

  public:
    struct storage_tag {};
    using value_t = T;
    using variance_t = T;

    explicit static_storage(std::size_t n=0) : data_(n * sizeof(T)) {}

    static_storage(const static_storage<T>& other) :
      data_(other.data_)
    {}

    static_storage(static_storage<T>&& other) :
      data_(std::move(other.data_))
    {}

    static_storage& operator=(static_storage<T>&& other)
    {
      if (this != &other) {
        data_ = std::move(other.data_);
      }
      return *this;
    }

    template <typename OtherStorage,
              typename = typename OtherStorage::storage_tag>
    static_storage(const OtherStorage& other) :
      data_(other.size() * sizeof(T))
    {
      for (std::size_t i = 0, n = size(); i < n; ++i)
        data_.at<T>(i) = other.value(i);
    }

    template <typename OtherStorage,
              typename = typename OtherStorage::storage_tag>
    static_storage(OtherStorage&& other) :
      data_(other.size() * sizeof(T))
    {
      for (std::size_t i = 0, n = size(); i < n; ++i)
        data_.at<T>(i) = other.value(i);
      other = OtherStorage();
    }

    template <typename OtherStorage,
              typename = typename OtherStorage::storage_tag>
    static_storage& operator=(const OtherStorage& other)
    {
      if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
        data_.resize(other.size() * sizeof(T));
        for (std::size_t i = 0, n = size(); i < n; ++i)
          data_.at<T>(i) = other.value(i);        
      }
      return *this;
    }

    template <typename OtherStorage,
              typename = typename OtherStorage::storage_tag>
    static_storage& operator=(OtherStorage&& other)
    {
      if (static_cast<void*>(this) != static_cast<void*>(&other)) {
        data_.resize(other.size() * sizeof(T));
        for (std::size_t i = 0, n = size(); i < n; ++i)
          data_.at<T>(i) = other.value(i);        
        other = OtherStorage();
      }
      return *this;
    }

    std::size_t size() const { return data_.nbytes() / sizeof(T); }
    constexpr unsigned depth() const { return sizeof(T); }
    const void* data() const { return data_.data(); }
    void increase(std::size_t i) { ++(data_.at<T>(i)); }
    value_t value(std::size_t i) const { return data_.at<T>(i); }
    variance_t variance(std::size_t i) const { return data_.at<T>(i); }

    template <typename OtherStorage,
              typename = typename OtherStorage::storage_tag>
    void operator+=(const OtherStorage& other)
    {
      for (std::size_t i = 0, n = size(); i < n; ++i)
        data_.at<T>(i) += other.value(i);
    }

  private:
    buffer_t data_;
 
    friend class dynamic_storage;
  };

}
}

#endif
