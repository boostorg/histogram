// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_BASE_HPP_
#define _BOOST_HISTOGRAM_AXIS_BASE_HPP_

#include <boost/config.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/utility/string_view.hpp>
#include <stdexcept>
#include <string>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
} // namespace serialization
} // namespace boost

namespace boost {
namespace histogram {
namespace axis {

enum class uoflow { off = 0, oflow = 1, on = 2 };

/// Base class for all axes
class base {
public:
  /// Returns the number of bins without overflow/underflow.
  int size() const noexcept { return size_; }
  /// Returns the number of bins, including overflow/underflow if enabled.
  int shape() const noexcept { return shape_; }
  /// Returns true if axis has extra overflow and underflow bins.
  bool uoflow() const noexcept { return shape_ > size_; }
  /// Returns the axis label, which is a name or description.
  string_view label() const noexcept { return label_; }
  /// Change the label of an axis.
  void label(string_view label) { label_.assign(label.begin(), label.end()); }

protected:
  base(unsigned size, string_view label, axis::uoflow uo)
      : size_(size),
        shape_(size + static_cast<int>(uo)),
        label_(label.begin(), label.end()) {
    if (size_ == 0) { throw std::invalid_argument("bins > 0 required"); }
  }

  base() = default;
  base(const base&) = default;
  base& operator=(const base&) = default;
  base(base&& rhs)
      : size_(rhs.size_), shape_(rhs.shape_), label_(std::move(rhs.label_)) {
    rhs.size_ = 0;
    rhs.shape_ = 0;
  }
  base& operator=(base&& rhs) {
    if (this != &rhs) {
      size_ = rhs.size_;
      shape_ = rhs.shape_;
      label_ = std::move(rhs.label_);
      rhs.size_ = 0;
      rhs.shape_ = 0;
    }
    return *this;
  }

  bool operator==(const base& rhs) const noexcept {
    return size_ == rhs.size_ && shape_ == rhs.shape_ && label_ == rhs.label_;
  }

private:
  int size_ = 0, shape_ = 0;
  std::string label_;

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
}
}
}

#endif
