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
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

/// Base class for all axes
template <typename MetaData, option Options>
class base {
public:
  using metadata_type = MetaData;

  /// Returns the number of bins, without extra bins.
  int size() const noexcept { return size_meta_.first(); }
  /// Returns the options.
  constexpr option options() const noexcept { return Options; }
  /// Returns the metadata.
  metadata_type& metadata() noexcept { return size_meta_.second(); }
  /// Returns the metadata (const version).
  const metadata_type& metadata() const noexcept { return size_meta_.second(); }

  friend void swap(base& a, base& b) noexcept // ADL works with friend functions
  {
    using std::swap;
    swap(a.size_meta_, b.size_meta_);
  }

  template <class Archive>
  void serialize(Archive&, unsigned);

protected:
  base(unsigned n, metadata_type m) : size_meta_(n, std::move(m)) {
    if (size() == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("bins > 0 required"));
    const auto max_index = std::numeric_limits<int>::max() -
                           test(Options, option::underflow) -
                           test(Options, option::overflow);
    if (size() > max_index)
      BOOST_THROW_EXCEPTION(
          std::invalid_argument(detail::cat("bins <= ", max_index, " required")));
  }

  base() : size_meta_(0) {}

  bool operator==(const base& rhs) const noexcept {
    return size() == rhs.size() && detail::relaxed_equal(metadata(), rhs.metadata());
  }

private:
  detail::compressed_pair<int, metadata_type> size_meta_;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
