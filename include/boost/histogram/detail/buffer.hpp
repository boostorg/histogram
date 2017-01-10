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

namespace boost {
namespace histogram {
namespace detail {

    class buffer {
    public:

        buffer(std::size_t s, unsigned d) :
            size_(s),
            depth_(d),
            memory_(nullptr)
        {
            if (size_ * depth_) {
                memory_ = std::calloc(size_, depth_);        
                if (!memory_)
                    throw std::bad_alloc();                
            }
        }

        buffer() :
            size_(0),
            depth_(0),
            memory_(nullptr)
        {}

        buffer(const buffer& o) :
            size_(o.size_), depth_(o.depth_), memory_(nullptr)
        {
            realloc();
            std::copy(static_cast<char*>(o.memory_),
                      static_cast<char*>(o.memory_) + size_ * depth_,
                      static_cast<char*>(memory_));
        }

        buffer& operator=(const buffer& o)
        {
            if (this != &o) {
                if ((size_ * depth_) != (o.size_ * o.depth_)) {
                    size_ = o.size_;
                    depth_ = o.depth_;
                    realloc();
                }
                std::copy(static_cast<const char*>(o.memory_),
                          static_cast<const char*>(o.memory_) + size_,
                          static_cast<char*>(memory_));
            }
            return *this;
        }

        template <typename T, template<typename> class Storage>
        buffer(const Storage<T>& o) :
            size_(o.size()), depth_(sizeof(T)), memory_(nullptr)
        {
            realloc();
            std::copy(static_cast<const T*>(o.data()),
                      static_cast<const T*>(o.data()) + size_,
                      static_cast<T*>(memory_));
        }

        template <typename T, template<typename> class Storage>
        buffer& operator=(const Storage<T>& o) {
            size_ = o.size();
            depth_ = sizeof(T);
            realloc();
            std::copy(o.data_, o.data_ + size_,
                      static_cast<T*>(memory_));
            return *this;
        }

        buffer(buffer&& o) :
            size_(o.size_),
            depth_(o.depth_),
            memory_(o.memory_)
        {
            o.size_ = 0;
            o.depth_ = 0;
            o.memory_ = nullptr;
        }

        buffer& operator=(buffer&& o)
        {
            if (this != &o) {
                std::free(memory_);
                size_ = o.size_;
                depth_ = o.depth_;
                memory_ = o.memory_;
                o.size_ = 0;
                o.depth_ = 0;
                o.memory_ = nullptr;
            }
            return *this;
        }

        template <typename T, template<typename> class Storage>
        buffer(Storage<T>&& o) :
            size_(o.size_),
            depth_(sizeof(T)),
            memory_(const_cast<void*>(o.data_))
        {
            o.size_ = 0;
            o.data_ = nullptr;
        }

        template <typename T, template<typename> class Storage>
        buffer& operator=(Storage<T>&& o)
        {
            std::free(memory_);
            size_ = o.size_;
            depth_ = sizeof(T);
            memory_ = static_cast<void*>(o.data_);
            o.size_ = 0;
            o.data_ = nullptr;
            return *this;
        }

        ~buffer() { std::free(memory_); }

        std::size_t size() const { return size_; }

        unsigned depth() const { return depth_; }

        const void* data() const { return memory_; }

        bool operator==(const buffer& o) const {
            return size_ == o.size_ &&
                   depth_ == o.depth_ &&
                   std::equal(static_cast<char*>(memory_),
                              static_cast<char*>(memory_) + size_ * depth_,
                              static_cast<char*>(o.memory_));
        }

        void depth(unsigned d)
        {
            depth_ = d;
            realloc();
        }

        void realloc()
        {
            if (!memory_) {
                if (size_ * depth_ > 0)
                    memory_ = std::calloc(size_, depth_);
            }
            else {
                memory_ = std::realloc(memory_, size_ * depth_);
            }
            if (!memory_ && (size_ * depth_ > 0))            
                throw std::bad_alloc();
        }

        template <typename T>
        T* begin() { return static_cast<T*>(memory_); }

        template <typename T>
        T* end() { return static_cast<T*>(memory_) + size_; }

        template <typename T>
        const T* cbegin() const { return static_cast<const T*>(memory_); }

        template <typename T>
        const T* cend() const { return static_cast<const T*>(memory_) + size_; }

        template <typename T>
        T& at(std::size_t i) { return begin<T>()[i]; }

        template <typename T>
        const T& at(std::size_t i) const { return cbegin<T>()[i]; }

    private:
        std::size_t size_;
        unsigned depth_;
        void* memory_;

        template <class Archive>
        friend void serialize(Archive&, buffer&, unsigned);
    };

}
}
}

#endif
