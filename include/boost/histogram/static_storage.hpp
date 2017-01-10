// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STATIC_STORAGE_HPP_
#define _BOOST_HISTOGRAM_STATIC_STORAGE_HPP_

#include <cstdlib>

namespace boost {
namespace histogram {

  namespace detail { class buffer; }

  template <typename T>
  class static_storage {
  public:
    using value_t = T;
    using variance_t = T;

    static_storage() : size_(0), data_(nullptr) {}

    explicit static_storage(std::size_t n) :
      size_(n),
      data_(n > 0 ? static_cast<T*>(std::calloc(n, sizeof(T))) : nullptr)
    {}

    static_storage(const static_storage<T>& other) :
      size_(other.size_),
      data_(static_cast<T*>(std::malloc(size_ * sizeof(T))))
    {
      std::copy(other.data_, other.data_ + size_, data_);
    }

    template <typename OtherStorage,
              typename = typename OtherStorage::value_t>
    static_storage(const OtherStorage& other) :
      size_(other.size()),
      data_(static_cast<T*>(std::malloc(size_ * sizeof(T))))
    {
      for (std::size_t i = 0; i < size_; ++i)
        data_[i] = other.value(i);
    }

    template <typename OtherStorage,
              typename = typename OtherStorage::value_t>
    static_storage& operator=(const OtherStorage& other)
    {
      if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
        std::free(data_);
        size_ = other.size();
        data_ = static_cast<T*>(std::malloc(size_ * sizeof(T)));
        for (std::size_t i = 0; i < size_; ++i)
          data_[i] = other.value(i);        
      }
      return *this;
    }

    static_storage(static_storage<T>&& other) :
      static_storage()
    {
      std::swap(size_, other.size_);
      std::swap(data_, other.data_);
    }

    static_storage& operator=(static_storage<T>&& other)
    {
      if (this != &other) {
        std::free(data_);
        size_ = other.size_;
        data_ = other.data_;
        other.size_ = 0;
        other.data_ = nullptr;
      }
      return *this;
    }

    template <typename OtherStorage,
              typename = typename OtherStorage::value_t>
    static_storage(OtherStorage&& other) :
      static_storage(static_cast<const OtherStorage&>(other))
    {
      other = OtherStorage();
    }

    template <typename OtherStorage,
              typename = typename OtherStorage::value_t>
    static_storage& operator=(OtherStorage&& other)
    {
      std::free(data_);
      size_ = other.size();
      data_ = static_cast<T*>(std::malloc(size_ * sizeof(T)));
      for (std::size_t i = 0, n = size(); i < n; ++i)
        data_[i] = other.value(i);        
      other = OtherStorage();
      return *this;
    }

    ~static_storage() { std::free(data_); }

    std::size_t size() const { return size_; }
    constexpr unsigned depth() const { return sizeof(T); }
    const void* data() const { return static_cast<const void*>(data_); }
    void increase(std::size_t i) { ++(data_[i]); }
    value_t value(std::size_t i) const { return data_[i]; }
    variance_t variance(std::size_t i) const { return data_[i]; }

    template <typename OtherStorage,
              typename = typename OtherStorage::value_t>
    void operator+=(const OtherStorage& other)
    {
      for (std::size_t i = 0, n = size(); i < n; ++i)
        data_[i] += other.value(i);
    }

  private:
    std::size_t size_;
    T* data_;

    friend detail::buffer;

    template <typename Archive, typename U>
    friend void serialize(Archive&, static_storage<U>&, unsigned);
  };

}
}

#endif
