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

    inline
    std::string escape(const std::string& s) {
        std::string os;
        os.reserve(s.size() + 2);
        os += '\'';
        for (auto sit = s.begin(); sit != s.end(); ++sit) {
            if (*sit == '\'' && (sit == s.begin() || *(sit-1) != '\\'))
                os += "\\\'";
            else
                os += *sit;
        }
        os += '\'';
        return os;
    }

    class buffer_t {
    public:
        buffer_t() :
            memory_(nullptr),
            nbytes_(0)
        {}

        buffer_t(std::size_t nbytes) :
            memory_(std::calloc(nbytes, sizeof(char))),
            nbytes_(nbytes)
        {
            if (!memory_)
                throw std::bad_alloc();
        }

        buffer_t(const buffer_t& o) :
            memory_(std::malloc(o.nbytes_)),
            nbytes_(o.nbytes_)
        {
            if (!memory_)
                throw std::bad_alloc();
            std::copy(static_cast<char*>(o.memory_),
                      static_cast<char*>(o.memory_) + nbytes_,
                      static_cast<char*>(memory_));
        }

        buffer_t(buffer_t&& o) :
            memory_(o.memory_),
            nbytes_(o.nbytes_)
        {
            o.memory_ = nullptr;
            o.nbytes_ = 0;
        }

        ~buffer_t() { std::free(memory_); }

        buffer_t& operator=(const buffer_t& o)
        {
            if (this != &o) {
                if (nbytes_ != o.nbytes_)
                    *this = buffer_t(o); // clang says: don't move here
                else
                    std::copy(static_cast<char*>(o.memory_),
                              static_cast<char*>(o.memory_) + nbytes_,
                              static_cast<char*>(memory_));
            }
            return *this;
        }

        buffer_t& operator=(buffer_t&& o)
        {
            if (nbytes_)
                std::free(memory_);
            memory_ = o.memory_;
            nbytes_ = o.nbytes_;
            o.memory_ = nullptr;
            o.nbytes_ = 0;
            return *this;
        }

        void resize(std::size_t nbytes)
        {
            memory_ = std::realloc(memory_, nbytes);
            nbytes_ = nbytes;
            if (!memory_ && nbytes > 0)
                throw std::bad_alloc();
        }

        template <typename T>
        T* begin() { return static_cast<T*>(memory_); }

        template <typename T>
        T* end() { return static_cast<T*>(memory_ + nbytes_); }

        template <typename T>
        const T* cbegin() const { return static_cast<const T*>(memory_); }

        template <typename T>
        const T* cend() const { return static_cast<const T*>(memory_ + nbytes_); }

        template <typename T>
        T& get(std::size_t i) { return begin<T>()[i]; }

        template <typename T>
        const T& get(std::size_t i) const { return cbegin<T>()[i]; }

        bool operator==(const buffer_t& o) const {
            return nbytes_ == o.nbytes_ &&
                   std::memcmp(memory_, o.memory_, nbytes_) == 0;
        }

        std::size_t nbytes() const { return nbytes_; }

        const void* data() const { return memory_; }

    private:
        void* memory_;
        std::size_t nbytes_;

        template <class T, class Archive>
        friend void serialize_impl(Archive&, buffer_t&, unsigned);
    };

}
}
}

#endif
