// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DISPLAY_HPP
#define BOOST_HISTOGRAM_DISPLAY_HPP

#include <boost/histogram.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

namespace boost {
namespace histogram {
namespace detail {

const unsigned int histogram_width = 60;  // 60 characters
const double max_bin_coefficient = 0.95;  // 95% of histogram_width
const unsigned int precision = 1; //precision of upper and lower bounds

template <class Histogram>
std::ostream& get_lower_bound(std::ostream& os, 
                              typename indexed_range<const Histogram>::range_iterator ri) {
  os << std::fixed << std::setprecision(precision);
  os << ri->bin().lower();
  return os;
}

template <class Histogram>
std::ostream& get_upper_bound(std::ostream& os,
                              typename indexed_range<const Histogram>::range_iterator ri) {
  os << std::fixed << std::setprecision(precision);
  os << ri->bin().upper();
  return os;
}

template <typename Histogram>
std::ostream& get_value(std::ostream& out, 
                                 typename indexed_range<const Histogram>::range_iterator ri,
                                 const unsigned int column_width) {

  std::ostringstream tmp;
  tmp << std::defaultfloat << *(ri);
  out << std::left << std::setw(column_width) << tmp.str();
  return out;
}

template <class Histogram>
float get_lower_bound(typename indexed_range<const Histogram>::range_iterator ri) {
  return ri->bin().lower();
}

template <class Histogram>
float get_upper_bound(typename indexed_range<const Histogram>::range_iterator ri) {
  return ri->bin().upper();
}

template <class Histogram>
float get_value(typename indexed_range<const Histogram>::range_iterator ri) {
  return *ri;
}

template <typename Histogram>
std::ostream& get_label(std::ostream& out, 
                                  typename indexed_range<const Histogram>::range_iterator ri,
                                  const unsigned int column_width1,
                                  const unsigned int column_width2) {
  char parenthesis = ' ';
  if ( std::isfinite(ri->bin().upper()) )
    parenthesis = ')';
  else
    parenthesis = ']';

  out << '[' << std::right << std::setw(column_width1);
  get_lower_bound<Histogram>(out, ri); 
  out << ", ";
  out << std::right << std::setw(column_width2);
  get_upper_bound<Histogram>(out, ri);
  out << parenthesis;

  return out;
}

unsigned int getNumOfChars(float number) {
  int counter = 0;
  if(number < 0) {
    ++counter; // minus sign
    number *= -1;
  }

  if(number == std::numeric_limits<float>::infinity()){
    counter += 3;
    return counter;
  }

  if(number > 0 && number < 1)
    ++counter; // extra for 0.1 - 0.9 range
  
  if(floor(number) != number){
    ++counter; // dot
    number *= std::pow(10, precision);
  }

  while(number > 1){
    number /= 10;
    ++counter;
  }
  return counter;
}

template <typename Histogram>
unsigned int get_max_upperb_width(const Histogram& h) {
  auto data = indexed(h, coverage::all);
  unsigned int max_length = 0;
  unsigned int temp = 0;
  for (auto ri = data.begin(); ri != data.end(); ++ri) {
    temp = getNumOfChars(get_upper_bound<Histogram>(ri));
    if (temp > max_length) 
      max_length = temp; 
  }
  return max_length;
}

template <typename Histogram>
unsigned int get_max_lowerb_width(const Histogram& h) {
  auto data = indexed(h, coverage::all);
  unsigned int max_length = 0;
  unsigned int temp = 0;
  for (auto ri = data.begin(); ri != data.end(); ++ri) {
    temp = getNumOfChars(get_lower_bound<Histogram>(ri));
    if (temp > max_length) 
      max_length = temp; 
  }
  return max_length;
}

template <typename Histogram>
unsigned int get_max_value_width(const Histogram& h) {
  auto data = indexed(h, coverage::all);
  unsigned int max_length = 0;
  unsigned int temp = 0;
  for (auto ri = data.begin(); ri != data.end(); ++ri) {
    temp = getNumOfChars(get_value<Histogram>(ri));
    if (temp > max_length) 
      max_length = temp; 
  }
  return max_length;
}

std::ostream& draw_line(std::ostream& out, const unsigned int num, const char c = '*', bool complete = true) {
  unsigned int i = 0;
  for (; i < num; ++i) out << c;

  if (complete == true) {
    for (; i < histogram_width; ++i) out << ' ';
  }
  return out;
}

template <class Histogram>
unsigned int calculate_scale_f(typename indexed_range<const Histogram>::range_iterator ri,
                               const double& max_value) {
  
  const double longest_bin = max_bin_coefficient * histogram_width;
  double result = *ri * longest_bin / max_value;
  return std::lround(result);
}

template <typename Histogram>
std::ostream& get_histogram_line(std::ostream& out,                                  
                                 typename indexed_range<const Histogram>::range_iterator ri,
                                 const double& max_value) {
  
  const auto scaled_value = calculate_scale_f<Histogram>(ri, max_value);

  out << "|";
  draw_line(out, scaled_value);
  out << '|';
  return out;
}

std::ostream& get_external_line(std::ostream& out, const unsigned int labels_width) {
  draw_line(out, labels_width, ' ', false);
  out << " +";
  draw_line(out, histogram_width, '-');
  out << '+';
  return out;
}

template <class Histogram>
std::ostream& draw_histogram(std::ostream& out, 
                              const Histogram& h, 
                              const unsigned int u_bounds_width, 
                              const unsigned int l_bounds_width, 
                              const unsigned int values_width, 
                              const unsigned int hist_shift) {
  auto data = indexed(h, coverage::all);
  const auto max_v = *std::max_element(h.begin(), h.end());

  out << "\n";
  get_external_line(out, hist_shift); 
  out << "\n";

  for (auto it = data.begin(); it != data.end(); ++it) {
    out << "  ";
    get_label<Histogram>(out, it, u_bounds_width, l_bounds_width);
    out << "  ";
    get_value<Histogram>(out, it, values_width);
    out << " ";
    get_histogram_line<Histogram>(out, it, max_v);
    out << "\n";
  }
  get_external_line(out, hist_shift);
  out << "\n\n";
  
  return out;
}

template <class Histogram>
void display_histogram(std::ostream& out, const Histogram& h) {
  const auto additional_offset = 8; // 8 white characters
  const auto l_bounds_width = get_max_lowerb_width(h);
  const auto u_bounds_width = get_max_upperb_width(h);
  const auto values_width = get_max_value_width(h);
  const auto hist_shift = l_bounds_width + u_bounds_width + values_width + additional_offset;

  draw_histogram(out, h, u_bounds_width, l_bounds_width, values_width, hist_shift);
}
} // ns detail

namespace os {

// template <class Histogram>
// struct display_t {
//    const Histogram& histogram_;
//    unsigned terminal_width_;
// };

// template <class Histogram>
// auto display(const Histogram& h, unsigned terminal_width = 0) {
//    return display_t<Histogram>{h, terminal_width == 0 ? 60 : terminal_width};
// }

// template <class Histogram>
// std::ostream& operator<<(std::ostream& os, const display_t<Histogram>& dh) {
//    // call your main display function here
//    detail::display_histogram(os, dh.histogram_, dh.terminal_width_);
//    return os;
// }

template <class Histogram>
std::ostream& operator<<(std::ostream& out, const Histogram& h) {
  return out << detail::display_histogram(h);
}
} // ns os

} // ns histogram
} // ns boost
#endif
