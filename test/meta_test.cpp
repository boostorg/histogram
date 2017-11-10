#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/mpl/unique.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/greater.hpp>

using namespace boost;
using namespace boost::mpl;
namespace bhd = boost::histogram::detail;

int main() {
  typedef vector_c<int, 2, 1, 1, 3> numbers;
  typedef vector_c<int, 1, 2, 3> expected;
  using result = bhd::unique_sorted<numbers>;

  BOOST_MPL_ASSERT((equal<result, expected, equal_to<_, _>>));

  typedef vector_c<int, 1, 2> numbers2;
  typedef vector_c<int, 0, 3> expected2;
  using result2 = bhd::anti_indices<4, numbers2>;

  BOOST_MPL_ASSERT((equal<result2, expected2, equal_to<_, _>>));

  struct no_variance_method {
    using value_type = int;
  };
  struct variance_method {
    using value_type = int;
    value_type variance(std::size_t) const;
  };

  BOOST_TEST_EQ(typename bhd::has_variance_support<no_variance_method>::type(), false);
  BOOST_TEST_EQ(typename bhd::has_variance_support<variance_method>::type(), true);

  return boost::report_errors();
}