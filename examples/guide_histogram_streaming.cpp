//[ guide_histogram_streaming

#include <boost/histogram.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <iostream>

namespace bh = boost::histogram;

int main() {
    namespace axis = boost::histogram::axis;

    enum { A, B, C };

    auto h = bh::make_static_histogram(
        axis::regular<>(2, -1, 1, "regular1", axis::uoflow::off),
        axis::regular<double, axis::transform::log>(2, 1, 10, "regular2"),
        axis::circular<>(4, 0.1, 1.0, "polar"),
        axis::variable<>({-1, 0, 1}, "variable", axis::uoflow::off),
        axis::category<>({A, B, C}, "category"),
        axis::integer<>(-1, 1, "integer", axis::uoflow::off)
    );

    std::cout << h << std::endl;

    /* prints:

    histogram(
      regular(2, -1, 1, label='regular1', uoflow=False),
      regular_log(2, 1, 10, label='regular2'),
      circular(4, phase=0.1, perimeter=1, label='polar'),
      variable(-1, 0, 1, label='variable', uoflow=False),
      category(0, 1, 2, label='category'),
      integer(-1, 1, label='integer', uoflow=False),
    )

    */
}

//]
