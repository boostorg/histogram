// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_BASE_HPP
#define BOOST_HISTOGRAM_AXIS_BASE_HPP

#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/compressed_pair.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

/// Base class for all axes
template <typename MetaData>
class base {
  using metadata_type = MetaData;

public:
  /// Returns the number of bins, without extra bins.
  unsigned size() const noexcept { return size_meta_.first(); }
  /// Returns the options.
  option_type options() const noexcept { return opt_; }
  /// Returns the metadata.
  metadata_type& metadata() noexcept { return size_meta_.second(); }
  /// Returns the metadata (const version).
  const metadata_type& metadata() const noexcept { return size_meta_.second(); }

  friend void swap(base& a, base& b) noexcept // ADL works with friend functions
  {
    using std::swap;
    swap(a.size_meta_, b.size_meta_);
    swap(a.opt_, b.opt_);
  }

  template <class Archive>
  void serialize(Archive&, unsigned);

protected:
  base(unsigned n, metadata_type m, option_type opt)
      : size_meta_(n, std::move(m)), opt_(opt) {
    if (size() == 0) { throw std::invalid_argument("bins > 0 required"); }
    const auto max_index =
        static_cast<unsigned>(std::numeric_limits<int>::max() - static_cast<int>(opt_));
    if (size() > max_index)
      throw std::invalid_argument(detail::cat("bins <= ", max_index, " required"));
  }

  base() : size_meta_(0), opt_(option_type::none) {}
  base(const base&) = default;
  base& operator=(const base&) = default;
  base(base&& rhs) : size_meta_(std::move(rhs.size_meta_)), opt_(rhs.opt_) {}
  base& operator=(base&& rhs) {
    if (this != &rhs) {
      size_meta_ = std::move(rhs.size_meta_);
      opt_ = rhs.opt_;
    }
    return *this;
  }

  bool operator==(const base& rhs) const noexcept {
    return size() == rhs.size() && opt_ == rhs.opt_ &&
           detail::static_if<detail::is_equal_comparable<metadata_type>>(
               [&rhs](const auto& m) { return m == rhs.metadata(); },
               [](const auto&) { return true; }, metadata());
  }

private:
  detail::compressed_pair<int, metadata_type> size_meta_;
  option_type opt_;
}; // namespace axis

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
