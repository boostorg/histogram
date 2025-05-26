#ifndef BOOST_HISTOGRAM_MULTI_WEIGHT_HPP
#define BOOST_HISTOGRAM_MULTI_WEIGHT_HPP

#include <boost/core/nvp.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/detail/iterator_adaptor.hpp>
#include <boost/core/span.hpp>
#include <memory>
#include <algorithm>
#include <iostream>

namespace boost {
namespace histogram {

template <class T>
struct multi_weight_value : public boost::span<T> {
    using boost::span<T>::span;

    void operator()(const boost::span<T> values) {
        if (values.size() != this->size())
            throw std::runtime_error("size does not match");
        auto it = this->begin();
        for (const T& x : values)
            *it++ += x;
    }

    bool operator==(const boost::span<T> values) const {
        if (values.size() != this->size())
            return false;

        return std::equal(this->begin(), this->end(), values.begin());
    }
    bool operator!=(const boost::span<T> values) const {return !operator==(values);}
    void operator+=(const std::vector<T> values) {
        if (values.size() != this->size())
            throw std::runtime_error("size does not match");
        auto it = this->begin();
        for (const T& x : values)
            *it++ += x;
    }
    void operator+=(const boost::span<T> values) {
        if (values.size() != this->size())
            throw std::runtime_error("size does not match");
        auto it = this->begin();
        for (const T& x : values)
            *it++ += x;
    }
    void operator=(const std::vector<T> values) {
        if (values.size() != this->size())
            throw std::runtime_error("size does not match");
        auto it = this->begin();
        for (const T& x : values)
            *it++ = x;
    }
};


template <class ElementType = double>
class multi_weight {
public:
    using element_type = ElementType;
    using value_type = multi_weight_value<element_type>;
    using reference = value_type;
    using const_reference = const value_type&;

    template <class Value, class Reference, class MWPtr>
    struct iterator_base : public detail::iterator_adaptor<iterator_base<Value, Reference, MWPtr>, std::size_t, Reference> {
        using base_type =  detail::iterator_adaptor<iterator_base<Value, Reference, MWPtr>, std::size_t, Reference>;

        iterator_base() = default;
        iterator_base(const iterator_base& other) : iterator_base(other.par_, other.base()) {}
        iterator_base(MWPtr par, std::size_t idx) : base_type{idx}, par_{par} {}

        Reference operator*() const {
            return par_->span_buffer_[this->base()];
        }

        MWPtr par_ = nullptr;
    };

    using iterator = iterator_base<value_type, reference, multi_weight*>;
    using const_iterator = iterator_base<const value_type, const_reference, const multi_weight*>;

    static constexpr bool has_threading_support() { return false; }

    multi_weight(const std::size_t k = 0) : nelem_{k} {}

    multi_weight(const multi_weight& other) : nelem_{other.nelem_} {
        reset(other.size_);
        std::copy(other.buffer_.get(), other.buffer_.get() + buffer_length_, buffer_.get());
    }

    multi_weight& operator=(const multi_weight& other) {
        nelem_ = other.nelem_;
        reset(other.size_);
        std::copy(other.buffer_.get(), other.buffer_.get() + buffer_length_, buffer_.get());
        return *this;
    }


    std::size_t size() const { return size_; }

    void reset(std::size_t n) {
        size_ = n;
        buffer_length_ = n * nelem_;
        buffer_.reset(new element_type[buffer_length_]);
        default_fill();
        span_buffer_.reset(new value_type[size_]);
        std::size_t i = 0;
        std::generate_n(span_buffer_.get(), size_, [&] () {
            auto tmp_span = value_type{buffer_.get() + i * nelem_, nelem_};
            i++;
            return tmp_span;
        });
    }

    template <class T = element_type, std::enable_if_t<!std::is_arithmetic<T>::value, bool> = true>
    void default_fill() {}

    template <class T = element_type, std::enable_if_t< std::is_arithmetic<T>::value, bool> = true>
    void default_fill() {
        std::fill_n(buffer_.get(), buffer_length_, 0);
    }

    iterator begin() { return {this, 0}; }
    iterator end() { return {this, size_}; }

    const_iterator begin() const { return {this, 0}; }
    const_iterator end() const { return {this, size_}; }

    reference operator[](std::size_t i) { return span_buffer_[i]; }
    const_reference operator[](std::size_t i) const { return span_buffer_[i]; }

    template <class T>
    bool operator==(const multi_weight<T>& other) const {
        if (buffer_length_ != other.buffer_length_)
            return false;
        return std::equal(buffer_.get(), buffer_.get() + buffer_length_, other.buffer_.get());
    }

    template <class T>
    bool operator!=(const multi_weight<T>& other) const { return !operator==(other); }

    template <class T>
    void operator+=(const multi_weight<T>& other) {
        if (buffer_length_ != other.buffer_length_) {
            throw std::runtime_error("size does not match");
        }
        for (std::size_t i = 0; i < buffer_length_; i++) {
            buffer_[i] += other.buffer_[i];
        }
    }


    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
        ar& make_nvp("size", size_);
        ar& make_nvp("nelem", nelem_);
        std::vector<element_type> w;
        if (Archive::is_loading::value)
        {
            ar& make_nvp("buffer", w);
            reset(size_);
            std::swap_ranges(buffer_.get(), buffer_.get() + buffer_length_, w.data());
        } else {
            w.assign(buffer_.get(), buffer_.get() + buffer_length_);
            ar& make_nvp("buffer", w);
        }
    }

public:
    std::size_t size_ = 0;
    std::size_t nelem_ = 0;
    std::size_t buffer_length_ = 0;
    std::unique_ptr<element_type[]> buffer_;
    std::unique_ptr<value_type[]> span_buffer_;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const multi_weight_value<T>& v) {
    os << "multi_weight_value(";
    bool first = true;
    for (const T& x : v)
        if (first) { first = false; os << x; }
        else os << ", " << x;
    os << ")";
    return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const multi_weight<T>& v) {
    os << "multi_weight(\n";
    int index = 0;
    for (const multi_weight_value<T>& x : v) {
        os << "Index " << index << ": " << x << "\n";
        index++;
    }
    os << ")";
    return os;
}

namespace algorithm {

/** Compute the sum over all histogram cells (underflow/overflow included by default).

  The implementation favors accuracy and protection against overflow over speed. If the
  value type of the histogram is an integral or floating point type,
  accumulators::sum<double> is used to compute the sum, else the original value type is
  used. Compilation fails, if the value type does not support operator+=. The return type
  is double if the value type of the histogram is integral or floating point, and the
  original value type otherwise.

  If you need a different trade-off, you can write your own loop or use `std::accumulate`:
  ```
  // iterate over all bins
  auto sum_all = std::accumulate(hist.begin(), hist.end(), 0.0);

  // skip underflow/overflow bins
  double sum = 0;
  for (auto&& x : indexed(hist))
    sum += *x; // dereference accessor

  // or:
  // auto ind = boost::histogram::indexed(hist);
  // auto sum = std::accumulate(ind.begin(), ind.end(), 0.0);
  ```

  @returns accumulator type or double

  @param hist Const reference to the histogram.
  @param cov  Iterate over all or only inner bins (optional, default: all).
*/
template <class A, class B>
std::vector<B> sum(const histogram<A, multi_weight<B>>& hist, const coverage cov = coverage::all) {
  using sum_type = typename histogram<A, multi_weight<B>>::value_type;
  // T is arithmetic, compute sum accurately with high dynamic range
  std::vector<B> v(unsafe_access::storage(hist).nelem_, 0.);
  sum_type sum(v);
  if (cov == coverage::all)
    for (auto&& x : hist) sum += x;
  else
    // sum += x also works if sum_type::operator+=(const sum_type&) exists
    for (auto&& x : indexed(hist)) sum += *x;
  return v;
}

} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif
