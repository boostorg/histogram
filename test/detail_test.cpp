// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/variant.hpp>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace boost::mpl;
using namespace boost::histogram::detail;

int main() {
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

  // cat
  { BOOST_TEST_EQ(cat("foo", 1, "bar"), std::string("foo1bar")); }

  // unique_sorted
  {
    typedef vector_c<int, 2, 1, 1, 3> numbers;
    typedef vector_c<int, 1, 2, 3> expected;
    using result = unique_sorted_t<numbers>;

    BOOST_MPL_ASSERT((equal<result, expected, equal_to<_, _>>));
  }

  // union
  {
    typedef vector<int, unsigned, char> main_vector;
    typedef vector<unsigned, void *> aux_vector;
    using result = union_t<main_vector, aux_vector>;

    typedef vector<int, unsigned, char, void *> expected;
    BOOST_MPL_ASSERT((equal<result, expected, std::is_same<_, _>>));
  }

  // has_variance_support
  {
    struct no_methods {};

    struct value_method {
      const double &value() const;
    };

    struct variance_method {
      const double &variance() const;
    };

    struct value_and_variance_methods {
      const double &value() const;
      const double &variance() const;
    };

    BOOST_TEST_EQ(typename has_variance_support<no_methods>::type(), false);
    BOOST_TEST_EQ(typename has_variance_support<value_method>::type(), false);
    BOOST_TEST_EQ(typename has_variance_support<variance_method>::type(),
                  false);
    BOOST_TEST_EQ(
        typename has_variance_support<value_and_variance_methods>::type(),
        true);
  }

  return boost::report_errors();
}
