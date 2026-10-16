#ifndef BOOST_HISTOGRAM_MULTI_CELL_HPP
#define BOOST_HISTOGRAM_MULTI_CELL_HPP

#include <algorithm>
#include <boost/core/nvp.hpp>
#include <boost/core/span.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/detail/iterator_adaptor.hpp>
#include <iostream>
#include <memory>
#include <valarray>

namespace boost {
namespace histogram {

namespace detail {
// CRTP
template <class Base>
struct multi_cell_mixin : public Base {
  using element_type = typename Base::value_type;

// multi_cell_value acts like an accumulator for multiple elemental values per bin
  void operator()(const boost::span<element_type> values) { operator+=(values); }

  template <class T>
  void operator+=(const boost::span<T> values) {
    check_size(values);
    auto it = this->begin();
    for (const T& x : values) *it++ += x;
  }

  template <class T>
  bool operator==(const boost::span<T> values) const {
    check_size(values);
    return std::equal(this->begin(), this->end(), values.begin());
  }

  template <class T>
  bool operator!=(const boost::span<T> values) const {
    return !operator==(values);
  }

  private:
    template <class T>
    bool check_size(const boost::span<T> values) const {
      if (values.size() != this->size()) {
        throw std::runtime_error("size does not match");
      }
      return true;
    }
};

} // namespace detail

template <class ElementType = double>
class multi_cell {
public:
  using element_type = ElementType;

  class value_type : public detail::multi_cell_mixin<std::valarray<element_type>> {
    multi_cell_value(boost::span<element_type> values) {
      this->assign(values.begin(), values.end());
    }
    multi_cell_value() = default;

    template <class S>
    void operator=(const S values) {
      this->assign(values.begin(), values.end());
    }
  };

  struct reference : public multi_cell_mixin<boost::span<element_type>> {

    template <class S>
    void operator=(const S values) {
      if (values.size() != this->size()) throw std::runtime_error("size does not match");
      auto it = this->begin();
      for (const T& x : values) *it++ = x;
    }
  };

  using reference = multi_cell_reference<element_type>;
  using const_reference = const reference;

  template <class Value, class Reference, class MWPtr>
  struct iterator_base
      : public detail::iterator_adaptor<iterator_base<Value, Reference, MWPtr>,
                                        std::size_t, Reference> {
    using base_type = detail::iterator_adaptor<iterator_base<Value, Reference, MWPtr>,
                                               std::size_t, Reference>;

    iterator_base() = default;
    iterator_base(const iterator_base& other) : iterator_base(other.par_, other.base()) {}
    iterator_base(MWPtr par, std::size_t idx) : base_type{idx}, par_{par} {}

    decltype(auto) operator*() const {
      return Reference{par_->buffer_.get() + this->base() * par_->nelem_, par_->nelem_};
    }

    MWPtr par_ = nullptr;
  };

  using iterator = iterator_base<value_type, reference, multi_cell*>;
  using const_iterator =
      iterator_base<const value_type, const_reference, const multi_cell*>;

  static constexpr bool has_threading_support() { return false; }

  multi_cell(const std::size_t k = 0) : nelem_{k} {}

  multi_cell(const multi_cell& other) { *this = other; }

  multi_cell& operator=(const multi_cell& other) {
    nelem_ = other.nelem_;
    reset(other.size_);
    std::copy(other.buffer_.get(), other.buffer_.get() + size_ * nelem_, buffer_.get());
    return *this;
  }

  std::size_t size() const { return size_; }

  void reset(std::size_t n) {
    size_ = n;
    buffer_.reset(new element_type[size_ * nelem_]);
    default_fill();
  }

  template <class T = element_type,
            std::enable_if_t<!std::is_arithmetic<T>::value, bool> = true>
  void default_fill() {}

  template <class T = element_type,
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
  bool operator==(const multi_cell<T>& other) const {
    if (size_ * nelem_ != other.size_ * other.nelem_) return false;
    return std::equal(buffer_.get(), buffer_.get() + size_ * nelem_, other.buffer_.get());
  }

  template <class T>
  bool operator!=(const multi_cell<T>& other) const {
    return !operator==(other);
  }

  template <class T>
  void operator+=(const multi_cell<T>& other) {
    if (size_ * nelem_ != other.size_ * other.nelem_) {
      throw std::runtime_error("size does not match");
    }
    for (std::size_t i = 0; i < size_ * nelem_; i++) { buffer_[i] += other.buffer_[i]; }
  }

  template <class Archive>
  void serialize(Archive& ar, unsigned /* version */) {
    ar& make_nvp("size", size_);
    ar& make_nvp("nelem", nelem_);
    std::vector<element_type> w;
    if (Archive::is_loading::value) {
      ar& make_nvp("buffer", w);
      reset(size_);
      std::swap_ranges(buffer_.get(), buffer_.get() + size_ * nelem_, w.data());
    } else {
      w.assign(buffer_.get(), buffer_.get() + size_ * nelem_);
      ar& make_nvp("buffer", w);
    }
  }

public:
  std::size_t size_ = 0;  // Number of bins
  std::size_t nelem_ = 0; // Number of weights per bin
  std::unique_ptr<element_type[]> buffer_;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const multi_cell_value<T>& v) {
  os << "multi_cell_value(";
  bool first = true;
  for (const T& x : v)
    if (first) {
      first = false;
      os << x;
    } else
      os << ", " << x;
  os << ")";
  return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const multi_cell_reference<T>& v) {
  os << "multi_cell_reference(";
  bool first = true;
  for (const T& x : v)
    if (first) {
      first = false;
      os << x;
    } else
      os << ", " << x;
  os << ")";
  return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const multi_cell<T>& v) {
  os << "multi_cell(\n";
  int index = 0;
  for (const multi_cell_reference<T>& x : v) {
    os << "Index " << index << ": " << x << "\n";
    index++;
  }
  os << ")";
  return os;
}

} // namespace histogram
} // namespace boost

#endif
