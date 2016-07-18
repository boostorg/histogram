// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_
#define _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_

#include <string>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <new> // for bad_alloc exception

namespace boost {
namespace histogram {
namespace detail {

    std::string escape(const std::string& s) {
        std::string os;
        os += '\'';
        for (unsigned i = 0, n = s.size(); i < n; ++i) {
            const char c = s[i];
            if (c == '\'' && (i == 0 || s[i - 1] != '\\'))
                os += "\\\'";
            else
                os += c;
        }
        os += '\'';
        return os;
    }

    class buffer_t {
    public:
        buffer_t() :
            data_(nullptr),
            size_(0)
        {}

        buffer_t(const buffer_t& o) :
            data_(std::malloc(o.size_)),
            size_(o.size_)
        {
            if (!data_)
                throw std::bad_alloc();
            std::copy(static_cast<unsigned char*>(o.data_),
                      static_cast<unsigned char*>(o.data_) + size_,
                      static_cast<unsigned char*>(data_));
        }

        buffer_t(buffer_t&& o) :
            data_(o.data_),
            size_(o.size_)
        {
            o.data_ = nullptr;
            o.size_ = 0;
        }

        ~buffer_t() { std::free(data_); }

        buffer_t& operator=(const buffer_t& o)
        {
            if (this != &o) {
                if (size_ != o.size_) {
                    std::free(data_);
                    data_ = std::malloc(o.size_);
                    size_ = o.size_;
                    if (!data_)
                        throw std::bad_alloc();
                }
                std::copy(static_cast<unsigned char*>(o.data_),
                          static_cast<unsigned char*>(o.data_) + size_,
                          static_cast<unsigned char*>(data_));
            }
            return *this;
        }

        buffer_t& operator=(buffer_t&& o)
        {
            if (data_)
                std::free(data_);
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
            return *this;
        }

        buffer_t(std::size_t n) :
            data_(std::calloc(n, 1)),
            size_(n)
        {
            if (!data_)
                throw std::bad_alloc();
        }

        void realloc(std::size_t n)
        {
            data_ = std::realloc(data_, n);
            size_ = n;
            if (!data_)
                throw std::bad_alloc();
        }

    template <typename T>
    T& get(std::size_t i) { return static_cast<T*>(data_)[i]; }

    template <typename T>
    const T& get(std::size_t i) const { return static_cast<T*>(data_)[i]; }

    bool operator==(const buffer_t& o) const {
        return size_ == o.size_ &&
               std::memcmp(data_, o.data_, size_) == 0;
    }

    std::size_t size() const { return size_; }

    private:
        void* data_;
        std::size_t size_;
    };
}
}
}

#endif
