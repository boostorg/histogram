#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/vector.hpp>
#include <type_traits>

using namespace boost;
using namespace boost::mpl;
using namespace boost::histogram::detail;

int main() {
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
    typedef vector<unsigned, void*> aux_vector;
    using result = union_t<main_vector, aux_vector>;

    typedef vector<int, unsigned, char, void*> expected;
    BOOST_MPL_ASSERT((equal<result, expected, std::is_same<_, _>>));
  }

  // has_variance_support
  {
    struct no_methods {};

    struct value_method {
      const double& value() const;
    };

    struct variance_method {
      const double& variance() const;
    };

    struct value_and_variance_methods {
      const double& value() const;
      const double& variance() const;
    };

    BOOST_TEST_EQ(typename has_variance_support<no_methods>::type(),
                  false);
    BOOST_TEST_EQ(typename has_variance_support<value_method>::type(),
                  false);
    BOOST_TEST_EQ(typename has_variance_support<variance_method>::type(),
                  false);
    BOOST_TEST_EQ(typename has_variance_support<value_and_variance_methods>::type(), true);
  }

  return boost::report_errors();
}