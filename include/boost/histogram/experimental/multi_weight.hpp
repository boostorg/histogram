#ifndef BOOST_HISTOGRAM_MULTI_WEIGHT_HPP
#define BOOST_HISTOGRAM_MULTI_WEIGHT_HPP

#include <algorithm>
#include <boost/core/nvp.hpp>
#include <boost/core/span.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/detail/iterator_adaptor.hpp>
#include <iostream>
#include <memory>

namespace boost {
namespace histogram {

template <class T, class BASE>
struct multi_weight_base : public BASE {
    using BASE::BASE;

    template <class S>
    bool operator==(const S values) const {
        if(values.size() != this->size())
            return false;

        return std::equal(this->begin(), this->end(), values.begin());
    }

    template <class S>
    bool operator!=(const S values) const {
        return !operator==(values);
    }
};

template <class T>
struct multi_weight_reference : public multi_weight_base<T, boost::span<T>> {
    // using boost::span<T>::span;
    using multi_weight_base<T, boost::span<T>>::multi_weight_base;

    void operator()(const boost::span<T> values) { operator+=(values); }

    // template <class S>
    // bool operator==(const S values) const {
    //     if(values.size() != this->size())
    //         return false;
    //
    //    return std::equal(this->begin(), this->end(), values.begin());
    //}
    //
    // template <class S>
    // bool operator!=(const S values) const {
    //    return !operator==(values);
    //}

    // void operator+=(const std::vector<T> values) {
    // operator+=(boost::span<T>(values)); }

    void operator+=(const boost::span<T> values) {
        // template <class S>
        // void operator+=(const S values) {
        if(values.size() != this->size())
            throw std::runtime_error("size does not match");
        auto it = this->begin();
        for(const T& x : values)
            *it++ += x;
    }

    template <class S>
    void operator=(const S values) {
        if(values.size() != this->size())
            throw std::runtime_error("size does not match");
        auto it = this->begin();
        for(const T& x : values)
            *it++ = x;
    }
};

template <class T>
struct multi_weight_value : public multi_weight_base<T, std::vector<T>> {
    using multi_weight_base<T, std::vector<T>>::multi_weight_base;

    multi_weight_value(const boost::span<T> values) {
        this->assign(values.begin(), values.end());
    }
    multi_weight_value() = default;

    void operator()(const boost::span<T> values) { operator+=(values); }

    // template <class S>
    // bool operator==(const S values) const {
    //     if(values.size() != this->size())
    //         return false;
    //
    //    return std::equal(this->begin(), this->end(), values.begin());
    //}
    //
    // template <class S>
    // bool operator!=(const S values) const {
    //    return !operator==(values);
    //}
    //
    // void operator+=(const std::vector<T> values) {
    // operator+=(boost::span<T>(values)); }

    // template <class S>
    // void operator+=(const S values) {
    void operator+=(const boost::span<T> values) {
        if(values.size() != this->size()) {
            if(this->size() > 0) {
                throw std::runtime_error("size does not match");
            }
            this->assign(values.begin(), values.end());
            return;
        }
        auto it = this->begin();
        for(const T& x : values)
            *it++ += x;
    }

    template <class S>
    void operator=(const S values) {
        this->assign(values.begin(), values.end());
    }
};

template <class ElementType = double>
class multi_weight {
  public:
    using element_type    = ElementType;
    using value_type      = multi_weight_value<element_type>;
    using reference       = multi_weight_reference<element_type>;
    using const_reference = const reference;

    template <class Value, class Reference, class MWPtr>
    struct iterator_base
        : public detail::iterator_adaptor<iterator_base<Value, Reference, MWPtr>,
                                          std::size_t,
                                          Reference> {
        using base_type
            = detail::iterator_adaptor<iterator_base<Value, Reference, MWPtr>,
                                       std::size_t,
                                       Reference>;

        iterator_base() = default;
        iterator_base(const iterator_base& other)
            : iterator_base(other.par_, other.base()) {}
        iterator_base(MWPtr par, std::size_t idx)
            : base_type{idx}
            , par_{par} {}

        decltype(auto) operator*() const {
            return Reference{par_->buffer_.get() + this->base() * par_->nelem_,
                             par_->nelem_};
        }

        MWPtr par_ = nullptr;
    };

    using iterator = iterator_base<value_type, reference, multi_weight*>;
    using const_iterator
        = iterator_base<const value_type, const_reference, const multi_weight*>;

    static constexpr bool has_threading_support() { return false; }

    multi_weight(const std::size_t k = 0)
        : nelem_{k} {}

    multi_weight(const multi_weight& other) { *this = other; }

    multi_weight& operator=(const multi_weight& other) {
        nelem_ = other.nelem_;
        reset(other.size_);
        std::copy(
            other.buffer_.get(), other.buffer_.get() + size_ * nelem_, buffer_.get());
        return *this;
    }

    std::size_t size() const { return size_; }

    void reset(std::size_t n) {
        size_ = n;
        buffer_.reset(new element_type[size_ * nelem_]);
        default_fill();
    }

    template <class T                                               = element_type,
              std::enable_if_t<!std::is_arithmetic<T>::value, bool> = true>
    void default_fill() {}

    template <class T                                              = element_type,
              std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
    void default_fill() {
        std::fill_n(buffer_.get(), size_ * nelem_, 0);
    }

    iterator begin() { return {this, 0}; }
    iterator end() { return {this, size_}; }

    const_iterator begin() const { return {this, 0}; }
    const_iterator end() const { return {this, size_}; }

    reference operator[](std::size_t i) {
        return reference{buffer_.get() + i * nelem_, nelem_};
    }
    const_reference operator[](std::size_t i) const {
        return const_reference{buffer_.get() + i * nelem_, nelem_};
    }

    template <class T>
    bool operator==(const multi_weight<T>& other) const {
        if(size_ * nelem_ != other.size_ * other.nelem_)
            return false;
        return std::equal(
            buffer_.get(), buffer_.get() + size_ * nelem_, other.buffer_.get());
    }

    template <class T>
    bool operator!=(const multi_weight<T>& other) const {
        return !operator==(other);
    }

    template <class T>
    void operator+=(const multi_weight<T>& other) {
        if(size_ * nelem_ != other.size_ * other.nelem_) {
            throw std::runtime_error("size does not match");
        }
        for(std::size_t i = 0; i < size_ * nelem_; i++) {
            buffer_[i] += other.buffer_[i];
        }
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
        ar& make_nvp("size", size_);
        ar& make_nvp("nelem", nelem_);
        std::vector<element_type> w;
        if(Archive::is_loading::value) {
            ar& make_nvp("buffer", w);
            reset(size_);
            std::swap_ranges(buffer_.get(), buffer_.get() + size_ * nelem_, w.data());
        } else {
            w.assign(buffer_.get(), buffer_.get() + size_ * nelem_);
            ar& make_nvp("buffer", w);
        }
    }

  public:
    std::size_t size_  = 0; // Number of bins
    std::size_t nelem_ = 0; // Number of weights per bin
    std::unique_ptr<element_type[]> buffer_;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const multi_weight_value<T>& v) {
    os << "multi_weight_value(";
    bool first = true;
    for(const T& x : v)
        if(first) {
            first = false;
            os << x;
        } else
            os << ", " << x;
    os << ")";
    return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const multi_weight_reference<T>& v) {
    os << "multi_weight_reference(";
    bool first = true;
    for(const T& x : v)
        if(first) {
            first = false;
            os << x;
        } else
            os << ", " << x;
    os << ")";
    return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const multi_weight<T>& v) {
    os << "multi_weight(\n";
    int index = 0;
    for(const multi_weight_reference<T>& x : v) {
        os << "Index " << index << ": " << x << "\n";
        index++;
    }
    os << ")";
    return os;
}

} // namespace histogram
} // namespace boost

#endif
