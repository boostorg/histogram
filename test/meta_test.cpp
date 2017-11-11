#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/vector_c.hpp>
#include <type_traits>

using namespace boost;
using namespace boost::mpl;
using namespace boost::histogram::detail;

int main() {
  {
    typedef vector_c<int, 2, 1, 1, 3> numbers;
    typedef vector_c<int, 1, 2, 3> expected;
    using result = unique_sorted<numbers>;

    BOOST_MPL_ASSERT((equal<result, expected, equal_to<_, _>>));
  }

  struct no_variance_method {
    using value_type = int;
  };
  struct variance_method {
    using value_type = int;
    value_type variance(std::size_t) const;
  };

  BOOST_TEST_EQ(typename has_variance_support<no_variance_method>::type(), false);
  BOOST_TEST_EQ(typename has_variance_support<variance_method>::type(), true);

  return boost::report_errors();
}