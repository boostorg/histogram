#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/collector.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <string>
#include "str.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram;

int main() {
  accumulators::collector<int, std::vector<int>> acc;
  BOOST_TEST_EQ(acc.count(), 0);
  BOOST_TEST_EQ(acc.value().size(), 0);
  BOOST_TEST_EQ(acc, accumulators::collector<int>{});
  BOOST_TEST_EQ(str(acc), "collector(0 entries)");

  acc(1);
  acc(2);
  acc(3);

  BOOST_TEST_EQ(acc.count(), 3);
  BOOST_TEST_EQ(acc.value().size(), 3);
  BOOST_TEST_EQ(acc.value()[0], 1.);
  BOOST_TEST_EQ(acc.value()[1], 2);
  BOOST_TEST_EQ(acc.value()[2], 3);
  BOOST_TEST_EQ(acc, acc);
  BOOST_TEST_EQ(str(acc), "collector(3 entries)");

  accumulators::collector<int, std::vector<int>> acc_copy(acc);
  BOOST_TEST_EQ(acc_copy.count(), 3);
  BOOST_TEST_EQ(acc_copy.value().size(), 3);
  acc_copy(4);
  BOOST_TEST_EQ(acc_copy.count(), 4);
  BOOST_TEST_NE(acc, acc_copy);

  {
    std::vector<int> v{10, 20, 30};
    accumulators::collector<int, std::vector<int>> acc_from_vector(v);
    BOOST_TEST_EQ(acc_from_vector.count(), 3);
    acc_from_vector(40);
    BOOST_TEST_EQ(acc_from_vector.count(), 4);
    BOOST_TEST_EQ(v.size(), 3);
  }

  {
    std::vector<int> v{10, 20, 30};
    accumulators::collector<int, std::vector<int>> acc_from_vector(std::move(v));
    BOOST_TEST_EQ(acc_from_vector.count(), 3);
    acc_from_vector(40);
    BOOST_TEST_EQ(acc_from_vector.count(), 4);
    BOOST_TEST_EQ(v.size(), 0);
  }

  {
    accumulators::collector<int, std::vector<int>> acc_from_vector(std::move(std::vector<int>{10, 20, 30}));
    BOOST_TEST_EQ(acc_from_vector.count(), 3);
    acc_from_vector(40);
    BOOST_TEST_EQ(acc_from_vector.count(), 4);
  }

  // by default template arguments are <double, std::vector<double>>
  BOOST_TEST_EQ(accumulators::collector{} += accumulators::collector{},
                accumulators::collector{});

  {
    accumulators::collector<int, std::vector<int>> acc2;
    acc += acc2; // acc = [1, 2, 3] acc2 = []
    BOOST_TEST_EQ(acc.count(), 3);
    acc2(100);   // acc2 = [100]
    acc += acc2; // acc = [1, 2, 3, 100] acc2 = [100]
    BOOST_TEST_EQ(acc.count(), 4);
    acc(200); // acc = [1, 2, 3, 100, 200] acc2 = [100]
    BOOST_TEST_EQ(acc.count(), 5);
    acc2(300);   // acc = [1, 2, 3, 100, 200] acc2 = [100, 300]
    acc += acc2; // acc = [1, 2, 3, 100, 200, 100, 300] acc2 = [100, 300]
    BOOST_TEST_EQ(acc.count(), 7);
    BOOST_TEST_EQ(acc2.count(), 2);
    BOOST_TEST_EQ(acc.value()[0], 1.);
    BOOST_TEST_EQ(acc.value()[3], 100);
    BOOST_TEST_EQ(acc.value()[4], 200);
    BOOST_TEST_EQ(acc.value()[5], 100);
    BOOST_TEST_EQ(acc.value()[6], 300);
  }

  return boost::report_errors();
}
