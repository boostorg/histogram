// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_TINY_STRING_HPP_
#define _BOOST_HISTOGRAM_DETAIL_TINY_STRING_HPP_

#include <cstring>
#include <memory>
#include <ostream>
#include <boost/operators.hpp>

namespace boost {
namespace histogram {
namespace detail {

class tiny_string :
    boost::operators<tiny_string>
{
public:
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

    tiny_string(const std::string& other) :
        tiny_string(other.c_str())
    {}

    tiny_string& operator=(const std::string& other) {
        tiny_string tmp(other.c_str());
        swap(tmp);
        return *this;
    }

    tiny_string(const char* other)
    {
        if (other) {
            const auto n = std::strlen(other) + 1;
            if (n > 1) {
                ptr_.reset(new char[n]);
                std::strcpy(ptr_.get(), other);
            }
        }
    }

    tiny_string& operator=(const char* other) {
        tiny_string tmp(other);
        swap(tmp);
        return *this;
    }

    bool operator==(const tiny_string& other) const {
        return std::strcmp(c_str(), other.c_str()) == 0;
    }

    void swap(tiny_string& other) noexcept {
        ptr_.swap(other.ptr_);
    }

    std::size_t size() const { return ptr_ ? std::strlen(ptr_.get()) : 0; }
    const char* c_str() const { return ptr_ ? ptr_.get() : ""; }
private:
    std::unique_ptr<char[]> ptr_;

    template <typename Archiv>
    friend void serialize(Archiv&, tiny_string&, unsigned);
};

inline void swap(tiny_string& a, tiny_string& b) { a.swap(b); }

inline std::ostream& operator<<(std::ostream& os, const tiny_string& s)
{ os << s.c_str(); return os; }

}
}
}

#endif
