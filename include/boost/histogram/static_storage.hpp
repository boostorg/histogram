// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STATIC_STORAGE_HPP_
#define _BOOST_HISTOGRAM_STATIC_STORAGE_HPP_

#include <boost/histogram/detail/utility.hpp>

namespace boost {
namespace histogram {

  template <typename T>
  class static_storage {
    using buffer_t = detail::buffer_t;

  public:
    using value_t = T;
    using variance_t = T;

    static_storage(std::size_t n=0) : data_(n * sizeof(T)) {}

    std::size_t size() const { return data_.nbytes() / sizeof(T); }
    constexpr unsigned depth() const { return sizeof(T); }
    void increase(std::size_t i) { ++(data_.get<T>(i)); }
    value_t value(std::size_t i) const { return data_.get<T>(i); }
    variance_t variance(std::size_t i) const { return data_.get<T>(i); }
    bool operator==(const static_storage& other) const
    { return data_ == other.data_; }
    void operator+=(const static_storage& other)
    {
      for (std::size_t i = 0, n = size(); i < n; ++i)
        data_.get<T>(i) += other.data_.get<T>(i);
    }

  private:
    buffer_t data_;
  };

}
}

#endif
