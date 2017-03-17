// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/static_histogram.hpp>
#include <boost/histogram/storage/container_storage.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/utility.hpp>
#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <limits>
#include <sstream>
#include <vector>

int main() {
    using namespace boost::histogram;
    namespace mpl = boost::mpl;

    // init_0
    {
        auto h = static_histogram<
          mpl::vector<integer_axis>,
          adaptive_storage<>
        >();
        BOOST_TEST_EQ(h.dim(), 1);
        BOOST_TEST_EQ(h.size(), 0);
        auto h2 = static_histogram<
          mpl::vector<integer_axis>,
          container_storage<std::vector<unsigned>>
        >();
        BOOST_TEST(h2 == h);
        auto h3 = static_histogram<
          mpl::vector<regular_axis<>>,
          adaptive_storage<>
        >();
        BOOST_TEST(!(h3 == h));
    }

    // init_1
    {
        auto h = make_static_histogram_with<adaptive_storage<>>(regular_axis<>{3, -1, 1});
        BOOST_TEST_EQ(h.dim(), 1);
        BOOST_TEST_EQ(h.size(), 5);
        BOOST_TEST_EQ(shape(h.axis<0>()), 5);
        auto h2 = make_static_histogram_with<container_storage<std::vector<unsigned>>>(regular_axis<>{3, -1, 1});
        BOOST_TEST(h2 == h);
    }

    // init_2
    {
        auto h = make_static_histogram_with<adaptive_storage<>>(regular_axis<>{3, -1, 1},
                                                             integer_axis{-1, 1});
        BOOST_TEST_EQ(h.dim(), 2);
        BOOST_TEST_EQ(h.size(), 25);
        BOOST_TEST_EQ(shape(h.axis<0>()), 5);
        BOOST_TEST_EQ(shape(h.axis<1>()), 5);
        auto h2 = make_static_histogram_with<container_storage<std::vector<unsigned>>>(regular_axis<>{3, -1, 1},
                                                                  integer_axis{-1, 1});
        BOOST_TEST(h2 == h);
    }

    // init_3
    {
        auto h = make_static_histogram_with<adaptive_storage<>>(regular_axis<>{3, -1, 1},
                                                             integer_axis{-1, 1},
                                                             circular_axis<>{3});
        BOOST_TEST_EQ(h.dim(), 3);
        BOOST_TEST_EQ(h.size(), 75);
        auto h2 = make_static_histogram_with<container_storage<std::vector<unsigned>>>(regular_axis<>{3, -1, 1},
                                                                  integer_axis{-1, 1},
                                                                  circular_axis<>{3});
        BOOST_TEST(h2 == h);
    }

    // init_4
    {
        auto h = make_static_histogram_with<adaptive_storage<>>(regular_axis<>{3, -1, 1},
                                                             integer_axis{-1, 1},
                                                             circular_axis<>{3},
                                                             variable_axis<>{-1, 0, 1});
        BOOST_TEST_EQ(h.dim(), 4);
        BOOST_TEST_EQ(h.size(), 300);
        auto h2 = make_static_histogram_with<container_storage<std::vector<unsigned>>>(regular_axis<>{3, -1, 1},
                                                                  integer_axis{-1, 1},
                                                                  circular_axis<>{3},
                                                                  variable_axis<>{-1, 0, 1});
        BOOST_TEST(h2 == h);
    }

    // init_5
    {
        auto h = make_static_histogram_with<adaptive_storage<>>(regular_axis<>{3, -1, 1},
                                                             integer_axis{-1, 1},
                                                             circular_axis<>{3},
                                                             variable_axis<>{-1, 0, 1},
                                                             category_axis{"A", "B", "C"});
        BOOST_TEST_EQ(h.dim(), 5);
        BOOST_TEST_EQ(h.size(), 900);
        auto h2 = make_static_histogram_with<container_storage<std::vector<unsigned>>>(regular_axis<>{3, -1, 1},
                                                                  integer_axis{-1, 1},
                                                                  circular_axis<>{3},
                                                                  variable_axis<>{-1, 0, 1},
                                                                  category_axis{"A", "B", "C"});
        BOOST_TEST(h2 == h);
    }

    // copy_ctor
    {
        auto h = make_static_histogram(integer_axis(0, 1),
                                       integer_axis(0, 2));
        h.fill(0, 0);
        auto h2 = decltype(h)(h);
        BOOST_TEST(h2 == h);
        auto h3 = static_histogram<
          mpl::vector<integer_axis, integer_axis>,
          container_storage<std::vector<unsigned>>
        >(h);
        BOOST_TEST(h3 == h);
    }

    // copy_assign
    {
        auto h = make_static_histogram(integer_axis(0, 1),
                                       integer_axis(0, 2));
        h.fill(0, 0);
        auto h2 = decltype(h)();
        BOOST_TEST(!(h == h2));
        h2 = h;
        BOOST_TEST(h == h2);
        // test self-assign
        h2 = h2;
        BOOST_TEST(h == h2);
        auto h3 = static_histogram<
          mpl::vector<integer_axis, integer_axis>,
          container_storage<std::vector<unsigned>>
        >();
        h3 = h;
        BOOST_TEST(h == h3);
    }

    // move
    {
        auto h = make_static_histogram(integer_axis(0, 1),
                                       integer_axis(0, 2));
        h.fill(0, 0);
        const auto href = h;
        decltype(h) h2(std::move(h));
        BOOST_TEST_EQ(h.dim(), 2);
        BOOST_TEST_EQ(h.sum(), 0);
        BOOST_TEST_EQ(h.size(), 0);
        BOOST_TEST(h2 == href);
        decltype(h) h3;
        h3 = std::move(h2);
        BOOST_TEST_EQ(h2.dim(), 2);
        BOOST_TEST_EQ(h2.sum(), 0);
        BOOST_TEST_EQ(h2.size(), 0);
        BOOST_TEST(h3 == href);
    }

    // equal_compare
    {
        auto a = make_static_histogram(integer_axis(0, 1));
        auto b = make_static_histogram(integer_axis(0, 1), integer_axis(0, 2));
        BOOST_TEST(!(a == b));
        BOOST_TEST(!(b == a));
        auto c = make_static_histogram(integer_axis(0, 1));
        BOOST_TEST(!(b == c));
        BOOST_TEST(!(c == b));
        BOOST_TEST(a == c);
        BOOST_TEST(c == a);
        auto d = make_static_histogram(regular_axis<>(2, 0, 1));
        BOOST_TEST(!(c == d));
        BOOST_TEST(!(d == c));
        c.fill(0);
        BOOST_TEST(!(a == c));
        BOOST_TEST(!(c == a));
        a.fill(0);
        BOOST_TEST(a == c);
        BOOST_TEST(c == a);
        a.fill(0);
        BOOST_TEST(!(a == c));
        BOOST_TEST(!(c == a));
    }

    // d1
    {
        auto h = make_static_histogram(integer_axis(0, 1));
        h.fill(0);
        h.fill(0);
        h.fill(-1);
        h.fill(10);

        BOOST_TEST_EQ(h.dim(), 1);
        BOOST_TEST_EQ(bins(h.axis<0>()), 2);
        BOOST_TEST_EQ(shape(h.axis<0>()), 4);
        BOOST_TEST_EQ(h.sum(), 4);

        BOOST_TEST_THROWS(h.value(-2), std::out_of_range);
        BOOST_TEST_EQ(h.value(-1), 1.0);
        BOOST_TEST_EQ(h.value(0), 2.0);
        BOOST_TEST_EQ(h.value(1), 0.0);
        BOOST_TEST_EQ(h.value(2), 1.0);
        BOOST_TEST_THROWS(h.value(3), std::out_of_range);

        BOOST_TEST_THROWS(h.variance(-2), std::out_of_range);
        BOOST_TEST_EQ(h.variance(-1), 1.0);
        BOOST_TEST_EQ(h.variance(0), 2.0);
        BOOST_TEST_EQ(h.variance(1), 0.0);
        BOOST_TEST_EQ(h.variance(2), 1.0);
        BOOST_TEST_THROWS(h.variance(3), std::out_of_range);
    }

    // d1_2
    {
        auto h = make_static_histogram(integer_axis(0, 1, "", false));
        h.fill(0);
        h.fill(-0);
        h.fill(-1);
        h.fill(10);

        BOOST_TEST_EQ(h.dim(), 1);
        BOOST_TEST_EQ(bins(h.axis<0>()), 2);
        BOOST_TEST_EQ(shape(h.axis<0>()), 2);
        BOOST_TEST_EQ(h.sum(), 2);

        BOOST_TEST_THROWS(h.value(-1), std::out_of_range);
        BOOST_TEST_EQ(h.value(0), 2.0);
        BOOST_TEST_EQ(h.value(1), 0.0);
        BOOST_TEST_THROWS(h.value(2), std::out_of_range);

        BOOST_TEST_THROWS(h.variance(-1), std::out_of_range);
        BOOST_TEST_EQ(h.variance(0), 2.0);
        BOOST_TEST_EQ(h.variance(1), 0.0);
        BOOST_TEST_THROWS(h.variance(2), std::out_of_range);
    }

    // d1w
    {
        auto h = make_static_histogram(regular_axis<>(2, -1, 1));
        h.fill(0);
        h.wfill(2.0, -1.0);
        h.fill(-1.0);
        h.fill(-2.0);
        h.wfill(5.0, 10.0);

        BOOST_TEST_EQ(h.sum(), 10);

        BOOST_TEST_EQ(h.value(-1), 1.0);
        BOOST_TEST_EQ(h.value(0), 3.0);
        BOOST_TEST_EQ(h.value(1), 1.0);
        BOOST_TEST_EQ(h.value(2), 5.0);

        BOOST_TEST_EQ(h.variance(-1), 1.0);
        BOOST_TEST_EQ(h.variance(0), 5.0);
        BOOST_TEST_EQ(h.variance(1), 1.0);
        BOOST_TEST_EQ(h.variance(2), 25.0);
    }

    // d2
    {
        auto h = make_static_histogram(regular_axis<>(2, -1, 1),
                                       integer_axis(-1, 1, "", false));
        h.fill(-1, -1);
        h.fill(-1, 0);
        std::array<double, 2> ai = {{-1., -10.}};
        h.fill(ai);
        double in[2] = {-10., 0.};
        h.fill(in, in+2);

        BOOST_TEST_EQ(h.dim(), 2);
        BOOST_TEST_EQ(bins(h.axis<0>()), 2);
        BOOST_TEST_EQ(shape(h.axis<0>()), 4);
        BOOST_TEST_EQ(bins(h.axis<1>()), 3);
        BOOST_TEST_EQ(shape(h.axis<1>()), 3);
        BOOST_TEST_EQ(h.sum(), 3);

        BOOST_TEST_EQ(h.value(-1, 0), 0.0);
        BOOST_TEST_EQ(h.value(-1, 1), 1.0);
        BOOST_TEST_EQ(h.value(-1, 2), 0.0);

        BOOST_TEST_EQ(h.value(0, 0), 1.0);
        BOOST_TEST_EQ(h.value(0, 1), 1.0);
        BOOST_TEST_EQ(h.value(0, 2), 0.0);

        BOOST_TEST_EQ(h.value(1, 0), 0.0);
        BOOST_TEST_EQ(h.value(1, 1), 0.0);
        BOOST_TEST_EQ(h.value(1, 2), 0.0);

        BOOST_TEST_EQ(h.value(2, 0), 0.0);
        BOOST_TEST_EQ(h.value(2, 1), 0.0);
        BOOST_TEST_EQ(h.value(2, 2), 0.0);

        BOOST_TEST_EQ(h.variance(-1, 0), 0.0);
        BOOST_TEST_EQ(h.variance(-1, 1), 1.0);
        BOOST_TEST_EQ(h.variance(-1, 2), 0.0);

        BOOST_TEST_EQ(h.variance(0, 0), 1.0);
        BOOST_TEST_EQ(h.variance(0, 1), 1.0);
        BOOST_TEST_EQ(h.variance(0, 2), 0.0);

        BOOST_TEST_EQ(h.variance(1, 0), 0.0);
        BOOST_TEST_EQ(h.variance(1, 1), 0.0);
        BOOST_TEST_EQ(h.variance(1, 2), 0.0);

        BOOST_TEST_EQ(h.variance(2, 0), 0.0);
        BOOST_TEST_EQ(h.variance(2, 1), 0.0);
        BOOST_TEST_EQ(h.variance(2, 2), 0.0);
    }

    // d2w
    {
        auto h = make_static_histogram(regular_axis<>(2, -1, 1),
                                       integer_axis(-1, 1, "", false));
        h.fill(-1, 0);       // -> 0, 1
        h.wfill(10, -1, -1); // -> 0, 0
        h.wfill(5, -1, -10); // is ignored
        h.wfill(7, -10, 0);  // -> -1, 1

        BOOST_TEST_EQ(h.sum(), 18);

        BOOST_TEST_EQ(h.value(-1, 0), 0.0);
        BOOST_TEST_EQ(h.value(-1, 1), 7.0);
        BOOST_TEST_EQ(h.value(-1, 2), 0.0);

        BOOST_TEST_EQ(h.value(0, 0), 10.0);
        BOOST_TEST_EQ(h.value(0, 1), 1.0);
        BOOST_TEST_EQ(h.value(0, 2), 0.0);

        BOOST_TEST_EQ(h.value(1, 0), 0.0);
        BOOST_TEST_EQ(h.value(1, 1), 0.0);
        BOOST_TEST_EQ(h.value(1, 2), 0.0);

        BOOST_TEST_EQ(h.value(2, 0), 0.0);
        BOOST_TEST_EQ(h.value(2, 1), 0.0);
        BOOST_TEST_EQ(h.value(2, 2), 0.0);

        BOOST_TEST_EQ(h.variance(-1, 0), 0.0);
        BOOST_TEST_EQ(h.variance(-1, 1), 49.0);
        BOOST_TEST_EQ(h.variance(-1, 2), 0.0);

        BOOST_TEST_EQ(h.variance(0, 0), 100.0);
        BOOST_TEST_EQ(h.variance(0, 1), 1.0);
        BOOST_TEST_EQ(h.variance(0, 2), 0.0);

        BOOST_TEST_EQ(h.variance(1, 0), 0.0);
        BOOST_TEST_EQ(h.variance(1, 1), 0.0);
        BOOST_TEST_EQ(h.variance(1, 2), 0.0);

        BOOST_TEST_EQ(h.variance(2, 0), 0.0);
        BOOST_TEST_EQ(h.variance(2, 1), 0.0);
        BOOST_TEST_EQ(h.variance(2, 2), 0.0);
    }

    // d3w
    {
        auto h = make_static_histogram(integer_axis(0, 3),
                                       integer_axis(0, 4),
                                       integer_axis(0, 5));
        for (auto i = 0; i < bins(h.axis<0>()); ++i)
            for (auto j = 0; j < bins(h.axis<1>()); ++j)
                for (auto k = 0; k < bins(h.axis<2>()); ++k)
        {
            h.wfill(i+j+k, i, j, k);
        }

        for (auto i = 0; i < bins(h.axis<0>()); ++i)
            for (auto j = 0; j < bins(h.axis<1>()); ++j)
                for (auto k = 0; k < bins(h.axis<2>()); ++k)
            BOOST_TEST_EQ(h.value(i, j, k), i+j+k);
    }

    // add_1
    {
        auto a = make_static_histogram_with<
          adaptive_storage<>
        >(integer_axis(-1, 1));
        auto b = make_static_histogram_with<
          container_storage<std::vector<unsigned>>
        >(integer_axis(-1, 1));
        a.fill(-1);
        b.fill(1);
        auto c = a;
        c += b;
        BOOST_TEST_EQ(c.value(-1), 0);
        BOOST_TEST_EQ(c.value(0), 1);
        BOOST_TEST_EQ(c.value(1), 0);
        BOOST_TEST_EQ(c.value(2), 1);
        BOOST_TEST_EQ(c.value(3), 0);
        auto d = a;
        d += b;
        BOOST_TEST_EQ(d.value(-1), 0);
        BOOST_TEST_EQ(d.value(0), 1);
        BOOST_TEST_EQ(d.value(1), 0);
        BOOST_TEST_EQ(d.value(2), 1);
        BOOST_TEST_EQ(d.value(3), 0);
    }

    // add_2
    {
        auto a = make_static_histogram_with<
          adaptive_storage<>
        >(integer_axis(-1, 1));
        auto b = make_static_histogram_with<
          adaptive_storage<>
        >(integer_axis(-1, 1));

        a.fill(0);
        b.wfill(3, -1);
        auto c = a;
        c += b;
        BOOST_TEST_EQ(c.value(-1), 0);
        BOOST_TEST_EQ(c.value(0), 3);
        BOOST_TEST_EQ(c.value(1), 1);
        BOOST_TEST_EQ(c.value(2), 0);
        BOOST_TEST_EQ(c.value(3), 0);
        auto d = a;
        d += b;
        BOOST_TEST_EQ(d.value(-1), 0);
        BOOST_TEST_EQ(d.value(0), 3);
        BOOST_TEST_EQ(d.value(1), 1);
        BOOST_TEST_EQ(d.value(2), 0);
        BOOST_TEST_EQ(d.value(3), 0);
    }

    // add_3
    {
        auto a = make_static_histogram_with<
          container_storage<std::vector<char>>
        >(integer_axis(-1, 1));
        auto b = make_static_histogram_with<
          container_storage<std::vector<unsigned>>
        >(integer_axis(-1, 1));
        a.fill(-1);
        b.fill(1);
        auto c = a;
        c += b;
        BOOST_TEST_EQ(c.value(-1), 0);
        BOOST_TEST_EQ(c.value(0), 1);
        BOOST_TEST_EQ(c.value(1), 0);
        BOOST_TEST_EQ(c.value(2), 1);
        BOOST_TEST_EQ(c.value(3), 0);
        auto d = a;
        d += b;
        BOOST_TEST_EQ(d.value(-1), 0);
        BOOST_TEST_EQ(d.value(0), 1);
        BOOST_TEST_EQ(d.value(1), 0);
        BOOST_TEST_EQ(d.value(2), 1);
        BOOST_TEST_EQ(d.value(3), 0);
    }

    // bad_add
    {
        auto a = make_static_histogram(integer_axis(0, 1));
        auto b = make_static_histogram(integer_axis(0, 2));
        BOOST_TEST_THROWS(a += b, std::logic_error);
    }

    // bad_index
    {
        std::vector<int> index(1, 5);
        auto a = make_static_histogram(integer_axis(0, 1));
        BOOST_TEST_THROWS(a.value(5), std::out_of_range);
        BOOST_TEST_THROWS(a.value(index), std::out_of_range);
        BOOST_TEST_THROWS(a.variance(5), std::out_of_range);
        BOOST_TEST_THROWS(a.variance(index), std::out_of_range);
    }

    // doc_example_0
    {
        namespace bh = boost::histogram;
        // create 1d-histogram with 10 equidistant bins from -1.0 to 2.0,
        // with axis of histogram labeled as "x"
        auto h = bh::make_static_histogram(bh::regular_axis<>(10, -1.0, 2.0, "x"));

        // fill histogram with data
        h.fill(-1.5); // put in underflow bin
        h.fill(-1.0); // included in first bin, bin interval is semi-open
        h.fill(-0.5);
        h.fill(1.1);
        h.fill(0.3);
        h.fill(1.7);
        h.fill(2.0);  // put in overflow bin, bin interval is semi-open
        h.fill(20.0); // put in overflow bin
        h.wfill(5.0, 0.1); // fill with a weighted entry, weight is 5.0

        std::ostringstream os1;
        // access histogram counts
        const auto& a = h.axis<0>();
        for (int i = -1, n = bins(a) + 1; i < n; ++i) {
            os1 << "bin " << i
                << " x in [" << left(a, i) << ", " << right(a ,i) << "): "
                << h.value(i) << " +/- " << std::sqrt(h.variance(i))
                << "\n";
        }

        std::ostringstream os2;
        os2 << "bin -1 x in [-inf, -1): 1 +/- 1\n"
               "bin 0 x in [-1, -0.7): 1 +/- 1\n"
               "bin 1 x in [-0.7, -0.4): 1 +/- 1\n"
               "bin 2 x in [-0.4, -0.1): 0 +/- 0\n"
               "bin 3 x in [-0.1, 0.2): 5 +/- 5\n"
               "bin 4 x in [0.2, 0.5): 1 +/- 1\n"
               "bin 5 x in [0.5, 0.8): 0 +/- 0\n"
               "bin 6 x in [0.8, 1.1): 0 +/- 0\n"
               "bin 7 x in [1.1, 1.4): 1 +/- 1\n"
               "bin 8 x in [1.4, 1.7): 0 +/- 0\n"
               "bin 9 x in [1.7, 2): 1 +/- 1\n"
               "bin 10 x in [2, inf): 2 +/- 1.41421\n";

        BOOST_TEST_EQ(os1.str(), os2.str());
    }

    // histogram_serialization
    {
        auto a = make_static_histogram(regular_axis<>(3, -1, 1, "r"),
                                       circular_axis<>(4, 0.0, 1.0, "p"),
                                       variable_axis<>({0.1, 0.2, 0.3, 0.4, 0.5}, "v"),
                                       category_axis{"A", "B", "C"},
                                       integer_axis(0, 1, "i"));
        a.fill(0.5, 0.1, 0.25, 1, 0);
        std::string buf;
        {
            std::ostringstream os;
            boost::archive::text_oarchive oa(os);
            oa << a;
            buf = os.str();
        }
        auto b = decltype(a)();
        BOOST_TEST(!(a == b));
        {
            std::istringstream is(buf);
            boost::archive::text_iarchive ia(is);
            ia >> b;
        }
        BOOST_TEST(a == b);
    }

    return boost::report_errors();
}
