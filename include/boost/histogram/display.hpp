// Copyright (c) 2019 Przemyslaw Bartosik
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DISPLAY_HPP
#define BOOST_HISTOGRAM_DISPLAY_HPP

#include <boost/histogram.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/fwd.hpp>
#include <algorithm> // max_element
#include <cmath>     // floor, pow
#include <iomanip>   // setw
#include <iosfwd>
#include <iostream>  // cout
#include <limits>    // infinity

namespace boost {
namespace histogram {
namespace detail {

struct display_settings {
  const unsigned int default_width = 80;   // default witdth of visualization
  const unsigned int min_width = 40;       // min width of visualization
  const unsigned int max_width = 120;      // max width of visualization
  unsigned int histogram_width = 60;       // default graph width
  unsigned int histogram_shift = 0;        // labels and values width
  const double max_bin_coefficient = 0.95; // 95% of histogram_width
  const unsigned int bounds_prec = 1;      // precision of upper and lower bounds
  const unsigned int values_prec = 0;      // precision of values
  const unsigned int margin = 2;           // margin from left edge
} d_s;

template <class Histogram>
void stream_lower_bound(std::ostream& out,
                        typename indexed_range<const Histogram>::range_iterator ri,
                        const unsigned int l_bounds_width = 0) {
  if (l_bounds_width != 0)
    out << std::right << std::setw(l_bounds_width);

  out << std::fixed << std::setprecision(d_s.bounds_prec) << ri->bin().lower();
}

template <class Histogram>
void stream_upper_bound(std::ostream& out,
                        typename indexed_range<const Histogram>::range_iterator ri,
                        const unsigned int u_bounds_width = 0) {
  if (u_bounds_width != 0)
    out << std::right << std::setw(u_bounds_width);

  out << std::fixed << std::setprecision(d_s.bounds_prec) << ri->bin().upper();
}

template <class Histogram>
void stream_value(std::ostream& out,
                  typename indexed_range<const Histogram>::range_iterator ri,
                  const unsigned int column_width = 0) {
  if (column_width != 0)
    out << std::left << std::setw(column_width);

  out << std::fixed << std::setprecision(d_s.values_prec) << *(ri);
}

template <class Histogram>
void stream_label(std::ostream& out,
                  typename indexed_range<const Histogram>::range_iterator ri,
                  const unsigned int l_bounds_width,
                  const unsigned int u_bounds_width) {
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

unsigned int get_num_of_chars(std::ostream& out) {
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
    if (temp > max_length)
      max_length = temp;
  }
  return max_length;
}

void stream_line(std::ostream& out,
                 const unsigned int num,
                 const char c = '*',
                 bool complete = true) {
  unsigned int i = 0;
  for (; i < num; ++i) out << c;

  if (complete == true) { //|****<---->|
    for (; i < d_s.histogram_width; ++i) out << ' ';
  }
}

template <class Histogram>
unsigned int calculate_scale_factor(typename indexed_range<const Histogram>::range_iterator ri,
                                    const double& max_value) {
  double result = 0;
  if (max_value != 0) {
    const double longest_bin = d_s.max_bin_coefficient * d_s.histogram_width;
    result = *ri * longest_bin / max_value;
  }
  return std::lround(result);
}

template <class Histogram>
void stream_histogram_line(std::ostream& out,
                           typename indexed_range<const Histogram>::range_iterator ri,
                           const double& max_value) {

  const auto scaled_value = calculate_scale_factor<Histogram>(ri, max_value);

  out << "|";
  stream_line(out, scaled_value);
  out << '|';
}

void stream_external_line(std::ostream& out) {
  stream_line(out, d_s.histogram_shift, ' ', false);
  out << "+";
  stream_line(out, d_s.histogram_width, '-');
  out << '+';
}

template <class Histogram>
void draw_histogram(std::ostream& out,
                    const Histogram& h,
                    const unsigned int u_bounds_width,
                    const unsigned int l_bounds_width,
                    const unsigned int values_width) {
  auto data = indexed(h, coverage::all);
  const auto max_v = *std::max_element(h.begin(), h.end());

  out << "\n";
  stream_external_line(out);
  out << "\n";

  for (auto it = data.begin(); it != data.end(); ++it) {
    stream_line(out, d_s.margin, ' ', false);
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

unsigned int adjust_histogram_width(unsigned int terminal_width) {
  const auto frame = 2; // |  |

  if (terminal_width < d_s.min_width)
    terminal_width = d_s.min_width;
  else if (terminal_width > d_s.max_width)
    terminal_width = d_s.max_width;

  return terminal_width - frame - d_s.histogram_shift;
}

template <class Histogram>
void display_histogram(std::ostream& out,
                       const Histogram& h,
                       const unsigned int terminal_width = d_s.default_width) {

  const auto additional_offset = 7; // 7 white characters
  const auto l_bounds_width = get_max_width(h, stream_lower_bound<Histogram>);
  const auto u_bounds_width = get_max_width(h, stream_upper_bound<Histogram>);
  const auto values_width = get_max_width(h, stream_value<Histogram>);

  d_s.histogram_shift = l_bounds_width + u_bounds_width + values_width +
                        additional_offset + d_s.margin;
  d_s.histogram_width = adjust_histogram_width(terminal_width);

  draw_histogram(out, h, u_bounds_width, l_bounds_width, values_width);
}

template <class Histogram>
void old_style_ostream(std::ostream& os, const Histogram& h) {
  if (os.width() != 0)
    os.width(0); // ignore setw

  os << "\n";
  stream_line(os, d_s.margin, ' ', false);
  os << "histogram(";
  unsigned n = 0;
  h.for_each_axis([&](const auto& a) {
    if (h.rank() > 1) os << "\n  ";
    stream_line(os, d_s.margin, ' ', false);
    os << a;
    if (++n < h.rank()) os << ",";
  });
  os << (h.rank() > 1 ? "\n" : ")");
  stream_line(os, d_s.margin, ' ', false);
  os << ")\n";
}

template <class Histogram>
void new_style_ostream(std::ostream& os, const Histogram& h) {
  const auto exp_width = static_cast<unsigned int>(os.width());
  if (exp_width == 0)
    display_histogram(os, h);
  else {
    os.width(0); // reset
    display_histogram(os, h, exp_width);
  }
}

} // ns detail

template <typename CharT, typename Traits, typename A, typename S>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const histogram<A, S>& h) {

  if (h.rank() == 1)
    detail::new_style_ostream(os, h);
  else
    detail::old_style_ostream(os, h);

  return os;
}

} // ns histogram
} // ns boost
#endif
