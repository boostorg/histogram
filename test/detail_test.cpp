// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/weight_counter.hpp>
#include <boost/variant.hpp>
#include <cstring>
#include <sstream>
using namespace boost::histogram::detail;

namespace boost { namespace histogram { namespace detail {
std::ostream& operator<<(std::ostream& os, const weight_counter& w) {
  os << "[ " << w.w << ", " << w.w2 << "]";
  return os;
}
}}}

int main() {
  // weight_counter
  {
    BOOST_TEST_EQ(weight_counter(0), weight_counter());
    weight_counter w(1);
    BOOST_TEST_EQ(w, weight_counter(1));
    BOOST_TEST_NE(w, weight_counter());
    BOOST_TEST_EQ(1, w);
    BOOST_TEST_EQ(w, 1);
    BOOST_TEST_NE(2, w);
    BOOST_TEST_NE(w, 2);
    w += 2.0;
    BOOST_TEST_EQ(w, weight_counter(3, 5));
    // consistency: a weighted counter increased by weight 1 multiplied
    // by 2 must be the same as a weighted counter increased by weight 2
    weight_counter u(0);
    u += 1.0;
    u *= 2;
    BOOST_TEST_EQ(u, weight_counter(2, 4));
    weight_counter v(0);
    v += 2.0;
    BOOST_TEST_EQ(u, v);
  }

  // escape0
  {
    std::ostringstream os;
    escape(os, std::string("abc"));
    BOOST_TEST_EQ(os.str(), std::string("'abc'"));
  }

  // escape1
  {
    std::ostringstream os;
    escape(os, std::string("abc\n"));
    BOOST_TEST_EQ(os.str(), std::string("'abc\n'"));
  }

  // escape2
  {
    std::ostringstream os;
    escape(os, std::string("'abc'"));
    BOOST_TEST_EQ(os.str(), std::string("'\\\'abc\\\''"));
  }

  // assign_axis unreachable branch
  {
    using V1 = boost::variant<float>;
    using V2 = boost::variant<int>;
    V1 v1(1.0);
    V2 v2(2);
    boost::apply_visitor(assign_axis<V1>(v1), v2);
    BOOST_TEST_EQ(v1, V1(1.0));
    BOOST_TEST_EQ(v2, V2(2));
  }

  // index_mapper 1
  {
    std::vector<unsigned> n{{2, 2}};
    std::vector<bool> b{{true, false}};
    index_mapper m(std::move(n), std::move(b));
    BOOST_TEST_EQ(m.first, 0);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 1);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 2);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 3);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), false);
  }

  // index_mapper 2
  {
    std::vector<unsigned> n{{2, 2}};
    std::vector<bool> b{{false, true}};
    index_mapper m(std::move(n), std::move(b));
    BOOST_TEST_EQ(m.first, 0);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 1);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 2);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 3);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), false);
  }

  return boost::report_errors();
}
