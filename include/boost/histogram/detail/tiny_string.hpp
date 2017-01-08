// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_TINY_STRING_HPP_
#define _BOOST_HISTOGRAM_DETAIL_TINY_STRING_HPP_

#include <cstring>
#include <memory>

namespace boost {
namespace histogram {
namespace detail {

class tiny_string {
public:
    tiny_string(const char* s)
    {
        if (s) {
            const auto n = std::strlen(s) + 1;
            ptr_.reset(new char[n]);
            std::strcpy(ptr_.get(), s);            
        }
    }

    tiny_string() = default;

    tiny_string(const tiny_string& other) :
        tiny_string(other.c_str())
    {}

    tiny_string& operator=(const tiny_string& other) {
        tiny_string tmp = other;
        swap(tmp);
        return *this;
    }

    tiny_string(tiny_string&& other) noexcept = default;
    tiny_string& operator=(tiny_string&& other) = default;

    bool operator==(const tiny_string& other) const {
        if (ptr_ && other.ptr_)
            return std::strcmp(ptr_.get(), other.ptr_.get()) == 0;
        return ptr_ == other.ptr_;
    }

    void swap(tiny_string& other) noexcept {
        ptr_.swap(other.ptr_);
    }

    std::size_t size() const { return ptr_ ? std::strlen(ptr_.get()) : 0; }
    const char* c_str() const { return ptr_.get(); }
private:
    std::unique_ptr<char[]> ptr_;

    template <typename Archiv>
    friend void serialize(Archiv&, tiny_string&, unsigned);
};

inline void swap(tiny_string& a, tiny_string& b) {
    a.swap(b);
}

}
}
}

#endif
