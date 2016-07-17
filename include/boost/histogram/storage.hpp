// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_HPP_
#define _BOOST_HISTOGRAM_STORAGE_HPP_

#include <vector>

namespace boost {
namespace histogram {

  template <typename T>
  struct static_storage {
    std::vector<T> data_;
    typedef T value_t;
    typedef T variance_t;
    std::size_t size() const { return data_.size(); }
    void allocate(std::size_t n) { data_.resize(n, 0); }
    void increase(std::size_t i) { ++(data_[i]); }
    value_t value(std::size_t i) const { return data_[i]; }
    variance_t variance(std::size_t i) const { return data_[i]; }
    bool operator==(const static_storage& other) const
    { return data_ == other.data_; }
    void operator+=(const static_storage& other)
    {
      for (std::size_t i = 0, n = size(); i < n; ++i)
        data_[i] += other.data_[i];
    }
  };

}
}

#endif
