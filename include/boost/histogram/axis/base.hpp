// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_BASE_HPP
#define BOOST_HISTOGRAM_AXIS_BASE_HPP

#include <boost/config.hpp>
#include <boost/container/string.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/utility/string_view.hpp>
#include <stdexcept>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
} // namespace serialization
} // namespace boost

namespace boost {
namespace histogram {
namespace axis {

enum class uoflow_type { off = 0, oflow = 1, on = 2 };

/// Base class for all axes
class base {
public:
  /// Returns the number of bins without overflow/underflow.
  int size() const noexcept { return size_; }
  /// Returns the number of bins, including overflow/underflow if enabled.
  int shape() const noexcept { return shape_; }
  /// Returns number of extra bins to count under- or overflow.
  int uoflow() const noexcept { return shape_ - size_; }

protected:
  base(unsigned size, axis::uoflow_type uo)
      : size_(size), shape_(size + static_cast<int>(uo)) {
    if (size_ == 0) { throw std::invalid_argument("bins > 0 required"); }
  }

  base() = default;
  base(const base&) = default;
  base& operator=(const base&) = default;
  base(base&& rhs) : size_(rhs.size_), shape_(rhs.shape_) {
    rhs.size_ = 0;
    rhs.shape_ = 0;
  }
  base& operator=(base&& rhs) {
    if (this != &rhs) {
      size_ = rhs.size_;
      shape_ = rhs.shape_;
      rhs.size_ = 0;
      rhs.shape_ = 0;
    }
    return *this;
  }

  bool operator==(const base& rhs) const noexcept {
    return size_ == rhs.size_ && shape_ == rhs.shape_;
  }

private:
  int size_ = 0, shape_ = 0;

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

/// Base class with a label
template <typename Allocator>
class labeled_base : public base {
public:
  using allocator_type = Allocator;

  allocator_type get_allocator() const { return label_.get_allocator(); }

  /// Returns the axis label, which is a name or description.
  boost::string_view label() const noexcept { return label_; }
  /// Change the label of an axis.
  void label(boost::string_view label) { label_.assign(label.begin(), label.end()); }

  bool operator==(const labeled_base& rhs) const noexcept {
    return base::operator==(rhs) && label_ == rhs.label_;
  }

protected:
  labeled_base() = default;
  labeled_base(const labeled_base&) = default;
  labeled_base& operator=(const labeled_base&) = default;
  labeled_base(labeled_base&& rhs) = default;
  labeled_base& operator=(labeled_base&& rhs) = default;

  labeled_base(unsigned size, axis::uoflow_type uo, string_view label,
               const allocator_type& a)
      : base(size, uo), label_(label.begin(), label.end(), a) {}

private:
  boost::container::basic_string<char, std::char_traits<char>, allocator_type> label_;

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

/// Iterator mixin, uses CRTP to inject iterator logic into Derived.
template <typename Derived>
class iterator_mixin {
public:
  using const_iterator = iterator_over<Derived>;
  using const_reverse_iterator = reverse_iterator_over<Derived>;

  const_iterator begin() const noexcept {
    return const_iterator(*static_cast<const Derived*>(this), 0);
  }
  const_iterator end() const noexcept {
    return const_iterator(*static_cast<const Derived*>(this),
                          static_cast<const Derived*>(this)->size());
  }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(*static_cast<const Derived*>(this),
                                  static_cast<const Derived*>(this)->size());
  }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(*static_cast<const Derived*>(this), 0);
  }
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
