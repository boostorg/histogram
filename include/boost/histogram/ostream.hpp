// Copyright 2015-2019 Hans Dembinski
// Copyright (c) 2019 Przemyslaw Bartosik
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_OSTREAM_HPP
#define BOOST_HISTOGRAM_OSTREAM_HPP

#include <algorithm> // max_element
#include <boost/histogram.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/fwd.hpp>
#include <cmath>   // floor, pow
#include <iomanip> // setw
#include <iosfwd>
#include <iostream> // cout
#include <limits>   // infinity

/**
  \file boost/histogram/ostream.hpp

  A simple streaming operator for the histogram type. The text representation is
  rudimentary and not guaranteed to be stable between versions of Boost.Histogram. This
  header is not included by any other header and must be explicitly included to use the
  streaming operator.

  To you use your own, simply include your own implementation instead of this header.
 */

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED

namespace boost {
namespace histogram {
namespace detail {

namespace display_settings {
constexpr unsigned int default_width = 80; // default witdth of visualization
constexpr unsigned int min_width = 40;     // min width of visualization
constexpr unsigned int max_width = 120;    // max width of visualization
// FIXME not thread-safe
static unsigned int histogram_width = 60;    // default graph width
constexpr unsigned int histogram_shift = 0;  // labels and values width
constexpr double max_bin_coefficient = 0.95; // 95% of histogram_width
constexpr unsigned int bounds_prec = 1;      // precision of upper and lower bounds
constexpr unsigned int values_prec = 0;      // precision of values
constexpr unsigned int margin = 2;           // margin from left edge
} // namespace display_settings

template <class OStream, class Histogram>
void stream_lower_bound(OStream& out,
                        typename indexed_range<const Histogram>::range_iterator ri,
                        const unsigned int l_bounds_width = 0) {
  if (l_bounds_width != 0) out << std::right << std::setw(l_bounds_width);

  out << std::fixed << std::setprecision(display_settings::bounds_prec)
      << ri->bin().lower();
}

template <class OStream, class Histogram>
void stream_upper_bound(OStream& out,
                        typename indexed_range<const Histogram>::range_iterator ri,
                        const unsigned int u_bounds_width = 0) {
  if (u_bounds_width != 0) out << std::right << std::setw(u_bounds_width);

  out << std::fixed << std::setprecision(display_settings::bounds_prec)
      << ri->bin().upper();
}

template <class OStream, class Histogram>
void stream_value(OStream& out,
                  typename indexed_range<const Histogram>::range_iterator ri,
                  const unsigned int column_width = 0) {
  if (column_width != 0) out << std::left << std::setw(column_width);

  out << std::fixed << std::setprecision(display_settings::values_prec) << *(ri);
}

template <class OStream, class Histogram>
void stream_label(OStream& out,
                  typename indexed_range<const Histogram>::range_iterator ri,
                  const unsigned int l_bounds_width, const unsigned int u_bounds_width) {
  char parenthesis = ' ';
  if (std::isfinite(ri->bin().upper()))
    parenthesis = ')';
  else
    parenthesis = ']';

  out << '[';
  stream_lower_bound<Histogram>(out, ri, l_bounds_width);
  out << ", ";
  stream_upper_bound<Histogram>(out, ri, u_bounds_width);
  out << parenthesis;
}

template <class OStream>
unsigned int get_num_of_chars(OStream& out) {
  auto result = static_cast<unsigned int>(out.tellp());
  out.clear();
  out.seekp(0); // reset
  return result;
}

template <class Histogram, class Getter>
unsigned int get_max_width(const Histogram& h, const Getter& streamFnPtr) {
  auto data = indexed(h, coverage::all);
  unsigned int max_length = 0;
  unsigned int temp = 0;
  std::ostringstream os;
  for (auto ri = data.begin(); ri != data.end(); ++ri) {
    streamFnPtr(os, ri, 0);
    temp = get_num_of_chars(os);
    if (temp > max_length) max_length = temp;
  }
  return max_length;
}

template <class OStream>
void stream_line(OStream& out, const unsigned int num, const char c = '*',
                 bool complete = true) {
  unsigned int i = 0;
  for (; i < num; ++i) out << c;

  if (complete == true) { //|****<---->|
    for (; i < display_settings::histogram_width; ++i) out << ' ';
  }
}

template <class Histogram>
unsigned int calculate_scale_factor(
    typename indexed_range<const Histogram>::range_iterator ri, const double& max_value) {
  double result = 0;
  if (max_value != 0) {
    const double longest_bin =
        display_settings::max_bin_coefficient * display_settings::histogram_width;
    result = *ri * longest_bin / max_value;
  }
  return std::lround(result);
}

template <class OStream, class Histogram>
void stream_histogram_line(OStream& out,
                           typename indexed_range<const Histogram>::range_iterator ri,
                           const double& max_value) {

  const auto scaled_value = calculate_scale_factor<Histogram>(ri, max_value);

  out << "|";
  stream_line(out, scaled_value);
  out << '|';
}

template <class OStream>
void stream_external_line(OStream& out) {
  stream_line(out, display_settings::histogram_shift, ' ', false);
  out << "+";
  stream_line(out, display_settings::histogram_width, '-');
  out << '+';
}

template <class OStream, class Histogram>
void draw_histogram(OStream& out, const Histogram& h, const unsigned int u_bounds_width,
                    const unsigned int l_bounds_width, const unsigned int values_width) {
  auto data = indexed(h, coverage::all);
  const auto max_v = *std::max_element(h.begin(), h.end());

  out << "\n";
  stream_external_line(out);
  out << "\n";

  for (auto it = data.begin(); it != data.end(); ++it) {
    stream_line(out, display_settings::margin, ' ', false);
    stream_label<Histogram>(out, it, l_bounds_width, u_bounds_width);
    out << "  ";
    stream_value<Histogram>(out, it, values_width);
    out << " ";
    stream_histogram_line<Histogram>(out, it, max_v);
    out << "\n";
  }
  stream_external_line(out);
  out << "\n\n";
}

inline unsigned int adjust_histogram_width(unsigned int terminal_width) {
  const auto frame = 2; // |  |

  if (terminal_width < display_settings::min_width)
    terminal_width = display_settings::min_width;
  else if (terminal_width > display_settings::max_width)
    terminal_width = display_settings::max_width;

  return terminal_width - frame - display_settings::histogram_shift;
}

template <class OStream, class Histogram>
void display_histogram(
    OStream& out, const Histogram& h,
    const unsigned int terminal_width = display_settings::default_width) {

  const auto additional_offset = 7; // 7 white characters
  const auto l_bounds_width = get_max_width(h, stream_lower_bound<Histogram>);
  const auto u_bounds_width = get_max_width(h, stream_upper_bound<Histogram>);
  const auto values_width = get_max_width(h, stream_value<Histogram>);

  display_settings::histogram_shift = l_bounds_width + u_bounds_width + values_width +
                                      additional_offset + display_settings::margin;
  display_settings::histogram_width = adjust_histogram_width(terminal_width);

  draw_histogram(out, h, u_bounds_width, l_bounds_width, values_width);
}

template <class OStream, class Histogram>
void old_style_ostream(OStream& os, const Histogram& h) {
  os << "histogram(";
  h.for_each_axis([&](const auto& a) { os << "\n  " << a << ","; });
  std::size_t i = 0;
  for (auto&& x : h) os << "\n  " << i++ << ": " << x;
  os << (h.rank() ? "\n)" : ")");
  return os;
}

template <class OStream, class Histogram>
void new_style_ostream(OStream& os, const Histogram& h) {
  const auto exp_width = static_cast<unsigned int>(os.width());
  if (exp_width == 0)
    display_histogram(os, h);
  else {
    os.width(0); // reset
    display_histogram(os, h, exp_width);
  }
}

} // namespace detail

template <typename CharT, typename Traits, typename A, typename S>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const histogram<A, S>& h) {

  if (h.rank() == 1)
    detail::new_style_ostream(os, h);
  else
    detail::old_style_ostream(os, h);

  return os;
}

} // namespace histogram
} // namespace boost

#endif // BOOST_HISTOGRAM_DOXYGEN_INVOKED

#endif
