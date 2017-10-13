#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/mpl/unique.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/type_traits.hpp>

using namespace boost;
using namespace boost::mpl;
namespace bhd = boost::histogram::detail;

int main() {
  typedef vector_c<int, 2, 1, 1, 3> numbers;
  typedef vector_c<int, 1, 2, 3> expected;
  using result = bhd::unique_sorted<numbers>;

  BOOST_MPL_ASSERT((equal<result, expected, equal_to<_, _>>));
  return boost::report_errors();
}