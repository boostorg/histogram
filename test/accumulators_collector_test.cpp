// Copyright 2024 Ruggero Turra, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/core/lightweight_test.hpp>
#include <boost/core/span.hpp>
#include <boost/histogram/accumulators/collector.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/detail/accumulator_traits.hpp>
#include <boost/histogram/detail/chunk_vector.hpp>
#include <string>
#include <tuple>
#include <type_traits>
#include "ostream.hpp"
#include "str.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram;

namespace boost {
template <class T1, class T2>
bool operator==(const span<T1>& a, const std::vector<T2>& b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end());
}
} // namespace boost

int main() {
  using traits = detail::accumulator_traits<accumulators::collector<>>;
  static_assert(!traits::weight_support, "");
  static_assert(std::is_same<traits::args, std::tuple<const double&>>::value, "");

  {
    accumulators::collector<std::vector<int>> acc;
    BOOST_TEST_EQ(acc.count(), 0);
    BOOST_TEST_EQ(acc.size(), 0);
    BOOST_TEST_EQ(str(acc), "collector{}");
    acc(1);
    BOOST_TEST_EQ(str(acc), "collector{1}");
    acc(2);
    BOOST_TEST_EQ(str(acc), "collector{1, 2}");
    BOOST_TEST_EQ(acc.count(), 2);
    BOOST_TEST_EQ(acc.size(), 2);

    const std::vector<int> ref = {{1, 2}};
    BOOST_TEST_ALL_EQ(acc.begin(), acc.end(), ref.begin(), ref.end());
  }

  {
    accumulators::collector<std::vector<int>> acc{{1, 2}};
    accumulators::collector<std::vector<int>> copy(acc);
    BOOST_TEST_ALL_EQ(copy.begin(), copy.end(), acc.begin(), acc.end());
    copy(3);
    BOOST_TEST_EQ(copy.size(), 3);
    BOOST_TEST_NE(acc, copy);
  }

  {
    accumulators::collector<std::vector<int>> acc1;
    acc1(1);
    acc1(2);

    accumulators::collector<std::vector<int>> acc2;
    BOOST_TEST_NE(acc1, acc2); // acc2 is empty
    acc2(2);
    acc2(1);
    BOOST_TEST_EQ(str(acc2), "collector{2, 1}");
    BOOST_TEST_NE(acc1, acc2); // order matters

    accumulators::collector<std::vector<int>> acc3;
    acc3(1);
    acc3(2);
    BOOST_TEST_EQ(acc1, acc3);

    // comparison to another embedded container type
    accumulators::collector<std::vector<double>> acc4;
    acc4(1);
    acc4(2);
    BOOST_TEST_EQ(acc1, acc4);

    // comparison to another container type
    std::vector<double> arr = {{1.0, 2.0}};
    BOOST_TEST_EQ(acc1, arr);
  }

  {
    BOOST_TEST_EQ(accumulators::collector<>{} += accumulators::collector<>{},
                  accumulators::collector<>{});

    accumulators::collector<> acc1;
    acc1(1);
    acc1(2);
    accumulators::collector<> acc2;
    acc1 += acc2; // acc1 = [1, 2] acc2 = []
    BOOST_TEST_EQ(acc1.count(), 2);
    acc2(3);      // acc2 = [3]
    acc1 += acc2; // acc = [1, 2, 3] acc2 = [3]
    BOOST_TEST_EQ(acc1.count(), 3);
    BOOST_TEST_EQ(str(acc1), "collector{1, 2, 3}");
    acc1 += accumulators::collector<std::vector<double>>{{4.0, 5.0}};
    BOOST_TEST_EQ(acc1.count(), 5);
    BOOST_TEST_EQ(str(acc1), "collector{1, 2, 3, 4, 5}");
  }

  {
    using A = std::array<int, 2>;
    accumulators::collector<std::vector<A>> acc;
    acc(A{1, 2});
    BOOST_TEST_EQ(acc.size(), 1);
    BOOST_TEST_EQ(str(acc), "collector{[ 1 2 ]}");
    acc(A{3, 4});
    BOOST_TEST_EQ(str(acc), "collector{[ 1 2 ], [ 3 4 ]}");
    BOOST_TEST_EQ(acc, (std::vector<A>{A{1, 2}, A{3, 4}}));
  }

  {
    accumulators::collector<detail::chunk_vector<int>> acc(2);

    std::vector<int> x = {{1, 2}};
    acc(x);

    BOOST_TEST_EQ(acc.size(), 1);
    BOOST_TEST_EQ(acc[0], x);

    x = {{3, 4}};
    acc(x);

    BOOST_TEST_EQ(acc.size(), 2);
    std::vector<int> x2 = {{1, 2}};
    BOOST_TEST_EQ(acc[0], x2);
    BOOST_TEST_EQ(acc[1], x);
    BOOST_TEST_EQ(acc[1][0], 3);
    BOOST_TEST_EQ(acc[1][1], 4);
  }

  return boost::report_errors();
}
