// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_BASE_HPP
#define BOOST_HISTOGRAM_AXIS_BASE_HPP

#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/iterator/reverse_iterator.hpp>
#include <stdexcept>
#include <limits>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

/// Base class for all axes
template <typename MetaData>
class base
{
  using metadata_type = MetaData;
  struct data : metadata_type // empty base class optimization
  {
    int size = 0;
    option_type opt = option_type::none;

    data() = default;
    data(const data&) = default;
    data& operator=(const data&) = default;
    data(data&& rhs)
      : metadata_type(std::move(rhs))
      , size(rhs.size), opt(rhs.opt)
    { rhs.size = 0; rhs.opt = option_type::none; }
    data& operator=(data&& rhs) {
      if (this != &rhs) {
        metadata_type::operator=(std::move(rhs));
        size = rhs.size;
        opt = rhs.opt;
        rhs.size = 0;
        rhs.opt = option_type::none;
      }
      return *this;
    }
    data(const metadata_type& m, int s, option_type o)
    : metadata_type(m), size(s), opt(o) {}

    bool operator==(const data& rhs) const noexcept {
      return size == rhs.size && opt == rhs.opt &&
        equal_impl(detail::is_equal_comparable<metadata_type>(), rhs);
    }

    bool equal_impl(std::true_type, const metadata_type& rhs) const noexcept {
      return static_cast<const metadata_type&>(*this) == rhs;
    }

    bool equal_impl(std::false_type, const metadata_type&) const noexcept {
      return true;
    }
  };

public:
  /// Returns the number of bins, without extra bins.
  unsigned size() const noexcept { return data_.size; }
  /// Returns the options.
  option_type options() const noexcept { return data_.opt; }
  /// Returns the metadata.
  metadata_type& metadata() noexcept { return static_cast<metadata_type&>(data_); }
  /// Returns the metadata (const version).
  const metadata_type& metadata() const noexcept { return static_cast<const metadata_type&>(data_); }

  template <class Archive>
  void serialize(Archive&, unsigned);

  friend void swap(base& a, base& b) noexcept // ADL works with friend functions
  {
    std::swap(static_cast<metadata_type&>(a), static_cast<metadata_type&>(b));
    std::swap(a.data_.size, b.data_.size);
    std::swap(a.data_.opt, b.data_.opt);
  }

protected:
  base(unsigned size, const metadata_type& m, option_type opt)
      : data_(m, size, opt)
  {
    if (size == 0) { throw std::invalid_argument("bins > 0 required"); }
    const auto max_index = static_cast<unsigned>(std::numeric_limits<int>::max()
      - static_cast<int>(data_.opt));
    if (size > max_index)
      throw std::invalid_argument(
        detail::cat("bins <= ", max_index, " required")
      );
  }

  base() = default;

  bool operator==(const base& rhs) const noexcept {
    return data_ == rhs.data_;
  }

private:
  data data_;
};

/// Uses CRTP to inject iterator logic into Derived.
template <typename Derived>
class iterator_mixin {
public:
  using const_iterator = iterator_over<Derived>;
  using const_reverse_iterator = boost::reverse_iterator<const_iterator>;

  const_iterator begin() const noexcept {
    return const_iterator(*static_cast<const Derived*>(this), 0);
  }
  const_iterator end() const noexcept {
    return const_iterator(*static_cast<const Derived*>(this),
                          static_cast<const Derived*>(this)->size());
  }
  const_reverse_iterator rbegin() const noexcept {
    return boost::make_reverse_iterator(end());
  }
  const_reverse_iterator rend() const noexcept {
    return boost::make_reverse_iterator(begin());
  }
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
