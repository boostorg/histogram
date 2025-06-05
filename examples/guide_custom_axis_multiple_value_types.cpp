// Copyright 2025 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_custom_axis_multiple_value_types

#include <boost/histogram.hpp>
#include <cassert>
#include <vector>

using namespace boost::histogram;

struct fraction {
  int numerator;
  int denominator;
};

// We can use a converter type to accept multiple value types.
struct my_axis : axis::regular<double> {
  using base_type = axis::regular<double>;

  using base_type::regular; // inherit constructors

  struct converter_type {
    double val_;
    // put overloads to handle multiple data types here or use a template
    converter_type(const fraction& x)
        : val_{static_cast<double>(x.numerator) / x.denominator} {}
    converter_type(double x) : val_{x} {}
  };

  axis::index_type index(converter_type x) const {
    return base_type::index(x.val_);
  }
};

int main() {
  auto h = make_histogram(my_axis(4, 0.0, 1.0));

  h(fraction{1, 3}); // 0.3333
  h(0.8);

  std::vector<fraction> a = {
      {1, 5}, // 0.2
      {3, 5}, // 0.6
  };
  h.fill(a);

  std::vector<double> b = {0.2, 0.4};
  h.fill(b);

  assert(h.at(0) == 2); // 0.0 ... 0.25
  assert(h.at(1) == 2); // 0.25 ... 0.5
  assert(h.at(2) == 1); // 0.5 ... 0.75
  assert(h.at(3) == 1); // 0.75 ... 1.0
}

//]
