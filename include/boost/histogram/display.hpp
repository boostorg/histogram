// Copyright (c) 2019  pb
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DISPLAY_HPP
#define BOOST_HISTOGRAM_DISPLAY_HPP

#include <boost/histogram.hpp>
#include <algorithm> //max_element
#include <iomanip>   //setw
#include <iostream>  //cout
#include <sstream>   //ostringstream
#include <cmath>     //floor, pow
#include <limits>    //infinity

namespace boost {
namespace histogram {
namespace detail {

struct display_settings {
  const unsigned int default_max_width = 80;  // default witdth of visualization 
  unsigned int histogram_width = 60;          // default graph width
  unsigned int histogram_shift = 0;           // labels and values width 
  const double max_bin_coefficient = 0.95;    // 95% of histogram_width
  const unsigned int precision = 1;           // precision of upper and lower bounds
} d_s;

template <class Histogram>
std::ostream& stream_lower_bound(std::ostream& os,
                                 typename indexed_range<const Histogram>::range_iterator ri) {
  os << std::fixed << std::setprecision(d_s.precision);
  os << ri->bin().lower();
  return os;
}

template <class Histogram>
std::ostream& stream_upper_bound(std::ostream& os,
                                 typename indexed_range<const Histogram>::range_iterator ri) {
  os << std::fixed << std::setprecision(d_s.precision);
  os << ri->bin().upper();
  return os;
}

template <class Histogram>
std::ostream& stream_value(std::ostream& out,
                           typename indexed_range<const Histogram>::range_iterator ri,
                           const unsigned int column_width) {

  std::ostringstream tmp;
  tmp << std::defaultfloat << *(ri);
  out << std::left << std::setw(column_width) << tmp.str();
  return out;
}

template <class Histogram>
double get_lower_bound(typename indexed_range<const Histogram>::range_iterator ri) {
  return ri->bin().lower();
}

template <class Histogram>
double get_upper_bound(typename indexed_range<const Histogram>::range_iterator ri) {
  return ri->bin().upper();
}

template <class Histogram>
double get_value(typename indexed_range<const Histogram>::range_iterator ri) {
  return *ri;
}

template <class Histogram>
std::ostream& stream_label(std::ostream& out,
                           typename indexed_range<const Histogram>::range_iterator ri,
                           const unsigned int l_bounds_width,
                           const unsigned int u_bounds_width) {
  char parenthesis = ' ';
  if ( std::isfinite(ri->bin().upper()) )
    parenthesis = ')';
  else
    parenthesis = ']';

  out << '[' << std::right << std::setw(l_bounds_width);
  stream_lower_bound<Histogram>(out, ri);
  out << ", ";
  out << std::right << std::setw(u_bounds_width);
  stream_upper_bound<Histogram>(out, ri);
  out << parenthesis;

  return out;
}

unsigned int get_num_of_chars(double number) {
  unsigned int counter = 0;
  if(number < 0) {
    ++counter; // minus sign
    number *= -1;
  }

  if(number == std::numeric_limits<double>::infinity()){
    counter += 3;
    return counter;
  }

  if(number > 0 && number < 1)
    ++counter; // extra +, for 0.1 - 0.9 range
  
  if(std::floor(number) != number){
    ++counter; // dot
    number *= std::pow(10, d_s.precision);
  }

  while(number > 1){
    number /= 10;
    ++counter;
  }
  return counter;
}

template <class Histogram, class Getter>
unsigned int get_max_width(const Histogram& h, const Getter& fcn) {
  auto data = indexed(h, coverage::all);
  unsigned int max_length = 0;
  unsigned int temp = 0;
  for (auto ri = data.begin(); ri != data.end(); ++ri) {
    temp = get_num_of_chars(fcn(ri));
    if (temp > max_length) 
      max_length = temp; 
  }
  return max_length;
}

std::ostream& draw_line(std::ostream& out,
                        const unsigned int num,
                        const char c = '*',
                        bool complete = true) {
  unsigned int i = 0;
  for (; i < num; ++i) out << c;

  if (complete == true) { //|****<---->|
    for (; i < d_s.histogram_width; ++i) out << ' ';
  }
  return out;
}

template <class Histogram>
unsigned int calculate_scale_factor(typename indexed_range<const Histogram>::range_iterator ri,
                                    const double& max_value) {
  
  const double longest_bin = d_s.max_bin_coefficient * d_s.histogram_width;
  double result = *ri * longest_bin / max_value;
  return std::lround(result);
}

template <class Histogram>
std::ostream& stream_histogram_line(std::ostream& out,
                                    typename indexed_range<const Histogram>::range_iterator ri,
                                    const double& max_value) {
  
  const auto scaled_value = calculate_scale_factor<Histogram>(ri, max_value);

  out << "|";
  draw_line(out, scaled_value);
  out << '|';
  return out;
}

std::ostream& stream_external_line(std::ostream& out) {
  draw_line(out, d_s.histogram_shift, ' ', false);
  out << "+";
  draw_line(out, d_s.histogram_width, '-');
  out << '+';
  return out;
}

template <class Histogram>
std::ostream& draw_histogram(std::ostream& out,
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
    out << "  ";
    stream_label<Histogram>(out, it, l_bounds_width, u_bounds_width);
    out << "  ";
    stream_value<Histogram>(out, it, values_width);
    out << " ";
    stream_histogram_line<Histogram>(out, it, max_v);
    out << "\n";
  }
  stream_external_line(out);
  out << "\n\n";
  
  return out;
}

unsigned int adjust_histogram_width(const unsigned int terminal_width) {
  const auto frame = 2; // |  |
  return terminal_width - frame - d_s.histogram_shift;
}

template <class Histogram>
void display_histogram(std::ostream& out,
                       const Histogram& h,
                       const unsigned int terminal_width = d_s.default_max_width) {
  
  const auto additional_offset = 9; // 9 white characters
  const auto l_bounds_width = get_max_width(h, get_lower_bound<Histogram>);
  const auto u_bounds_width = get_max_width(h, get_upper_bound<Histogram>);
  const auto values_width = get_max_width(h, get_value<Histogram>);
  d_s.histogram_shift = l_bounds_width + u_bounds_width + values_width + additional_offset;
  d_s.histogram_width = adjust_histogram_width(terminal_width);

  draw_histogram(out, h, u_bounds_width, l_bounds_width, values_width);
}
} // ns detail


template <typename CharT, typename Traits, typename A, typename S>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const histogram<A, S>& h) {
  auto tmp = os.width();
  if (tmp == 0)
    detail::display_histogram(os, h);
  else {
    os.width(0);
    detail::display_histogram(os, h, tmp);
  }
  return os;
}

} // ns histogram
} // ns boost
#endif
