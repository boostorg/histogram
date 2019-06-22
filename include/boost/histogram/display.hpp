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
  std::vector<int> scale_factors_;

  size_t lower_bounds_width_ = 0;
  size_t upper_bounds_width_ = 0;
  size_t str_values_width_ = 0;
  size_t external_line_shift_ = 0;
  visualization_data(const std::vector<std::string>& str_values,
                     const std::vector<int>& scale_factors,
                     const size_t& lower_bounds_width, const size_t& upper_bounds_width,
                     const size_t& str_values_width, const size_t& external_line_shift)
      : str_values_{str_values}
      , scale_factors_{scale_factors}
      , lower_bounds_width_{lower_bounds_width}
      , upper_bounds_width_{upper_bounds_width}
      , str_values_width_{str_values_width}
      , external_line_shift_{external_line_shift} {}
};

template <class Histogram>
extract extract_data(const Histogram& h) {
  std::stringstream lower, upper;
  lower << std::fixed << std::setprecision(1);
  upper << std::fixed << std::setprecision(1);

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

std::ostream& get_single_label(std::ostream& label, 
                             const extract& data, 
                             const unsigned int index,
                             const unsigned int column_width1,
                             const unsigned int column_width2) {
  char parenthesis = ' ';
  std::string lower = data.lower_bounds_.at(index);
  std::string upper = data.upper_bounds_.at(index);
  if (index == data.size() - 1)
    parenthesis = ']';
  else
    parenthesis = ')';

  label << '[' << std::right << std::setw(column_width1) << lower << ", "
               << std::right << std::setw(column_width2) << upper << parenthesis;

  return label;
}

std::ostream& get_single_str_value(std::ostream& str_value, 
                                 const std::vector<std::string>& str_values,
                                 const unsigned int index,
                                 const unsigned int column_width) {

  str_value << std::left << std::setw(column_width) << str_values.at(index);
  return str_value;
}

std::vector<int> calculate_scale_factors(const std::vector<double>& values) {
  std::vector<int> scale_factors{};
  const double longest_bin = max_bin_coefficient * histogram_width;

  auto max_value = std::max_element(values.begin(), values.end());

  for (const auto& x : values) {
    double result = x * longest_bin / (*max_value);
    scale_factors.push_back(static_cast<int>(result));
  }
  return scale_factors;
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

std::ostream& draw_line(std::ostream& line, const unsigned int num, const char c = '*', bool complete = true) {
  unsigned int i = 0;
  for (; i < num; ++i) line << c;

  if (complete == true) {
    for (; i < histogram_width; ++i) line << ' ';
  }
  return line;
}

std::ostream& get_single_histogram_line(std::ostream& line, 
                                      const std::vector<int>& values,
                                      const unsigned int index) {

  line << "|";
  draw_line(line, values.at(index));
  line << '|';
  return line;
}

std::ostream& get_external_line(std::ostream& external_line, const unsigned int labels_width) {
  draw_line(external_line, labels_width, ' ', false);
  external_line << " +";
  draw_line(external_line, histogram_width, '-');
  external_line << '+';
  return external_line;
}

visualization_data precalculate_visual_data(extract& h_data) {
  const unsigned int additional_offset = 8; // 8 white characters
  const auto scale_factors = calculate_scale_factors(h_data.values_);
  const auto str_values = convert_to_str_vec(h_data.values_);
  const auto lower_width = get_max_width(h_data.lower_bounds_);
  const auto upper_width = get_max_width(h_data.upper_bounds_);
  const auto str_values_width = get_max_width(str_values);
  const auto hist_shift =
      lower_width + upper_width + str_values_width + additional_offset;

  visualization_data v_data(str_values, scale_factors, lower_width, upper_width,
                            str_values_width, hist_shift);
  return v_data;
}

std::ostream& draw_histogram(std::ostream& visualisation, const extract& h_data, const visualization_data& v_data) {
  visualisation << "\n";
  get_external_line(visualisation, v_data.external_line_shift_); 
  visualisation << "\n";

  for (unsigned int i = 0; i < h_data.size(); i++) {
    visualisation << "  ";
    get_single_label(visualisation, h_data, i, v_data.lower_bounds_width_, v_data.upper_bounds_width_);
    visualisation << "  ";
    get_single_str_value(visualisation, v_data.str_values_, i, v_data.str_values_width_);
    visualisation << " ";
    get_single_histogram_line(visualisation, v_data.scale_factors_, i);
    visualisation << "\n";
  }
  get_external_line(visualisation, v_data.external_line_shift_);
  visualisation << "\n\n";
  
  return visualisation;
}

template <class Histogram>
void display_histogram(std::ostream& out, const Histogram& h) {
  auto histogram_data = detail::extract_data(h);
  auto visualization_data = detail::precalculate_visual_data(histogram_data);

  draw_histogram(out, histogram_data, visualization_data);
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
