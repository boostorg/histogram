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
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/throw_exception.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

/// Base class for all axes
template <typename MetaData, option_type Options>
class base {
public:
  using metadata_type = MetaData;

  /// Returns the number of bins, without extra bins.
  unsigned size() const noexcept { return size_meta_.first(); }
  /// Returns the options.
  constexpr option_type options() const noexcept { return Options; }
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
    const auto max_index = static_cast<unsigned>(std::numeric_limits<int>::max() -
                                                 (options() & option_type::underflow) -
                                                 (options() & option_type::overflow));
    if (size() > max_index)
      BOOST_THROW_EXCEPTION(
          std::invalid_argument(detail::cat("bins <= ", max_index, " required")));
  }

  base() : size_meta_(0) {}

  bool operator==(const base& rhs) const noexcept {
    return size() == rhs.size() &&
           detail::static_if<detail::is_equal_comparable<metadata_type>>(
               [&rhs](const auto& m) { return m == rhs.metadata(); },
               [](const auto&) { return true; }, metadata());
  }

private:
  detail::compressed_pair<int, metadata_type> size_meta_;
}; // namespace axis

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
