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

namespace boost {
namespace histogram {
namespace detail {

const unsigned int histogram_width = 60;  // 60 characters
const double max_bin_coefficient = 0.95;  // 95% of histogram_width

struct extract {
  std::vector<std::string> upper_bounds_;
  std::vector<std::string> lower_bounds_;
  std::vector<double> values_;
  unsigned int size() const { return values_.size(); }
};

struct visualization_data {
  std::vector<std::string> str_values_;

  size_t lower_bounds_width_ = 0;
  size_t upper_bounds_width_ = 0;
  size_t str_values_width_ = 0;
  size_t external_line_shift_ = 0;
  visualization_data(const std::vector<std::string>& str_values,
                     const size_t& lower_bounds_width, const size_t& upper_bounds_width,
                     const size_t& str_values_width, const size_t& external_line_shift)
      : str_values_{str_values}
      , lower_bounds_width_{lower_bounds_width}
      , upper_bounds_width_{upper_bounds_width}
      , str_values_width_{str_values_width}
      , external_line_shift_{external_line_shift} {}
};


template <class Histogram>
std::ostream& get_lower_bound(std::ostream& os, 
                              typename indexed_range<const Histogram>::range_iterator ri,
                              unsigned int prec = 1) {
  os << std::fixed << std::setprecision(prec);
  os << ri->bin().lower();
  return os;
}

template <class Histogram>
std::ostream& get_upper_bound(std::ostream& os,
                              typename indexed_range<const Histogram>::range_iterator ri,
                              unsigned int prec = 1) {
  os << std::fixed << std::setprecision(prec);
  os << ri->bin().upper();
  return os;
}

template <typename Histogram>
extract extract_data(const Histogram& h) {
  std::stringstream lower, upper;
  lower << std::fixed << std::setprecision(1);
  upper << std::fixed << std::setprecision(1);

  //std::remove_reference_t<Histogram> a;
  auto data = indexed(h, coverage::all);

  extract ex;
  for (const auto& x : data) {
    lower << x.bin().lower();
    upper << x.bin().upper();
    ex.lower_bounds_.push_back(lower.str());
    ex.upper_bounds_.push_back(upper.str());
    ex.values_.push_back(*x);
    lower.str("");
    upper.str("");
  }
  return ex;
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

template <typename Histogram>
std::ostream& get_value(std::ostream& out, 
                                 typename indexed_range<const Histogram>::range_iterator ri,
                                 const unsigned int column_width) {

  std::ostringstream tmp;
  tmp << std::defaultfloat << *(ri);
  out << std::left << std::setw(column_width) << tmp.str();
  return out;
}

size_t get_max_width(const std::vector<std::string>& container) {
  size_t max_length = 0;

  for (const auto& line : container)
    if (line.length() > max_length) max_length = line.length();
  return max_length;
}

std::vector<std::string> convert_to_str_vec(const std::vector<double>& values) {
  std::vector<std::string> string_values;
  string_values.reserve(values.size());


  std::transform(values.begin(), values.end(), std::back_inserter(string_values),
                 [](double d) {
                    std::ostringstream tmp;
                    tmp << std::defaultfloat << d;
                    return tmp.str();
                   });
  return string_values;
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

visualization_data precalculate_visual_data(extract& h_data) {
  const unsigned int additional_offset = 8; // 8 white characters
  const auto str_values = convert_to_str_vec(h_data.values_);
  const auto lower_width = get_max_width(h_data.lower_bounds_);
  const auto upper_width = get_max_width(h_data.upper_bounds_);
  const auto str_values_width = get_max_width(str_values);
  const auto hist_shift =
      lower_width + upper_width + str_values_width + additional_offset;

  visualization_data v_data(str_values, lower_width, upper_width,
                            str_values_width, hist_shift);
  return v_data;
}

template <class Histogram>
std::ostream& draw_histogram(std::ostream& out, const Histogram& h, const visualization_data& v_data) {
  auto data = indexed(h, coverage::all);
  const auto max_v = *std::max_element(h.begin(), h.end());

  out << "\n";
  get_external_line(out, v_data.external_line_shift_); 
  out << "\n";

  for (auto it = data.begin(); it != data.end(); ++it) {
    out << "  ";
    get_label<Histogram>(out, it, v_data.lower_bounds_width_, v_data.upper_bounds_width_);
    out << "  ";
    get_value<Histogram>(out, it, v_data.str_values_width_);
    out << " ";
    get_histogram_line<Histogram>(out, it, max_v);
    out << "\n";
  }
  get_external_line(out, v_data.external_line_shift_);
  out << "\n\n";
  
  return out;
}

template <class Histogram>
void display_histogram(std::ostream& out, const Histogram& h) {
  auto histogram_data = detail::extract_data(h);
  auto visualization_data = detail::precalculate_visual_data(histogram_data);

  draw_histogram(out, h, visualization_data);
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
