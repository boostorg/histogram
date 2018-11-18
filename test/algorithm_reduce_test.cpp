// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/algorithm/reduce.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <vector>
#include "utility_histogram.hpp"

using namespace boost::histogram;
using namespace boost::histogram::algorithm;

template <typename Tag>
void run_tests() {
  using regular = axis::regular<axis::transform::identity<>, axis::null_type>;
  {
    auto h = make(Tag(), regular(4, 1, 5), regular(3, -1, 2));
    h(1, -1);
    h(1, 0);
    h(2, 0);
    h(2, 1);
    h(2, 1);
    h(3, -1);
    h(3, 1);
    h(4, 1);
    h(4, 1);
    h(4, 1);

    /*
      matrix layout:
      x ->
    y 1 0 1 0
    | 1 1 0 0
    v 0 2 1 3
    */

    // should do nothing, index order does not matter
    auto hr = reduce(h, shrink(1, -1, 2), rebin(0, 1));
    BOOST_TEST_EQ(hr.rank(), 2);
    BOOST_TEST_EQ(sum(hr), 10);
    BOOST_TEST_EQ(hr.axis(0).size(), h.axis(0).size());
    BOOST_TEST_EQ(hr.axis(1).size(), h.axis(1).size());
    BOOST_TEST_EQ(hr.axis(0)[0].lower(), 1);
    BOOST_TEST_EQ(hr.axis(0)[3].upper(), 5);
    BOOST_TEST_EQ(hr.axis(1)[0].lower(), -1);
    BOOST_TEST_EQ(hr.axis(1)[2].upper(), 2);
    BOOST_TEST_EQ(hr, h);

    // not allowed: repeated indices
    BOOST_TEST_THROWS(reduce(h, rebin(0, 2), rebin(0, 2)), std::invalid_argument);
    BOOST_TEST_THROWS(reduce(h, shrink(1, 0, 2), shrink(1, 0, 2)), std::invalid_argument);
    // not allowed: shrink with lower == upper
    BOOST_TEST_THROWS(reduce(h, shrink(0, 0, 0)), std::invalid_argument);
    // not allowed: shrink axis to zero size
    BOOST_TEST_THROWS(reduce(h, shrink(0, 10, 11)), std::invalid_argument);
    // not allowed: rebin with zero merge
    BOOST_TEST_THROWS(reduce(h, rebin(0, 0)), std::invalid_argument);

    hr = reduce(h, shrink(0, 2, 4));
    BOOST_TEST_EQ(hr.rank(), 2);
    BOOST_TEST_EQ(sum(hr), 10);
    BOOST_TEST_EQ(hr.axis(0).size(), 2);
    BOOST_TEST_EQ(hr.axis(1).size(), h.axis(1).size());
    BOOST_TEST_EQ(hr.axis(0)[0].lower(), 2);
    BOOST_TEST_EQ(hr.axis(0)[1].upper(), 4);
    BOOST_TEST_EQ(hr.axis(1)[0].lower(), -1);
    BOOST_TEST_EQ(hr.axis(1)[2].upper(), 2);
    BOOST_TEST_EQ(hr.at(-1, 0), 1); // underflow
    BOOST_TEST_EQ(hr.at(0, 0), 0);
    BOOST_TEST_EQ(hr.at(1, 0), 1);
    BOOST_TEST_EQ(hr.at(2, 0), 0); // overflow
    BOOST_TEST_EQ(hr.at(-1, 1), 1);
    BOOST_TEST_EQ(hr.at(0, 1), 1);
    BOOST_TEST_EQ(hr.at(1, 1), 0);
    BOOST_TEST_EQ(hr.at(2, 1), 0);
    BOOST_TEST_EQ(hr.at(-1, 2), 0);
    BOOST_TEST_EQ(hr.at(0, 2), 2);
    BOOST_TEST_EQ(hr.at(1, 2), 1);
    BOOST_TEST_EQ(hr.at(2, 2), 3);

    /*
      matrix layout:
      x ->
    y 1 0 1 0
    | 1 1 0 0
    v 0 2 1 3
    */

    hr = reduce(h, shrink_and_rebin(0, 2, 5, 2), rebin(1, 3));
    BOOST_TEST_EQ(hr.rank(), 2);
    BOOST_TEST_EQ(sum(hr), 10);
    BOOST_TEST_EQ(hr.axis(0).size(), 1);
    BOOST_TEST_EQ(hr.axis(1).size(), 1);
    BOOST_TEST_EQ(hr.axis(0)[0].lower(), 2);
    BOOST_TEST_EQ(hr.axis(0)[0].upper(), 4);
    BOOST_TEST_EQ(hr.axis(1)[0].lower(), -1);
    BOOST_TEST_EQ(hr.axis(1)[0].upper(), 2);
    BOOST_TEST_EQ(hr.at(-1, 0), 2); // underflow
    BOOST_TEST_EQ(hr.at(0, 0), 5);
    BOOST_TEST_EQ(hr.at(1, 0), 3); // overflow

    std::vector<reduce_option_type> opts{{shrink_and_rebin(0, 2, 5, 2), rebin(1, 3)}};
    auto hr2 = reduce(h, opts);
    BOOST_TEST_EQ(hr2, hr);
  }

  // rebin on integer axis must fail
  {
    auto h = make(Tag(), axis::integer<>(1, 4));
    BOOST_TEST_THROWS(reduce(h, rebin(0, 2)), std::invalid_argument);
  }

  // reduce on axis with inverted range
  {
    auto h = make(Tag(), regular(4, 2, -2));
    auto hr = reduce(h, shrink(0, 1, -1));
    BOOST_TEST_EQ(hr.axis().size(), 2);
    BOOST_TEST_EQ(hr.axis()[0].lower(), 1);
    BOOST_TEST_EQ(hr.axis()[1].upper(), -1);
  }

  // reduce does not work with arguments not convertible to double
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
