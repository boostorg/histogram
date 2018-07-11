//[ getting_started_listing_01

#include <boost/histogram.hpp>
#include <iostream>

int main(int, char**) {
    namespace bh = boost::histogram;
    using namespace bh::literals; // enables _c suffix

    /*
      create a static 1d-histogram with an axis that has 6 equidistant
      bins on the real line from -1.0 to 2.0, and label it as "x"
    */
    auto h = bh::make_static_histogram(
      bh::axis::regular<>(6, -1.0, 2.0, "x")
    );

    // fill histogram with data, typically this happens in a loop
    // STL algorithms are supported
    auto data = { -0.5, 1.1, 0.3, 1.7 };
    std::for_each(data.begin(), data.end(), h);

    /*
      a regular axis is a sequence of semi-open bins; extra under- and
      overflow bins extend the axis in the default configuration
      index   :     -1    0    1   2   3   4   5   6
      bin edge:  -inf -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 inf
    */
    h(-1.5); // put in underflow bin -1
    h(-1.0); // put in bin 0, bin interval is semi-open
    h(2.0);  // put in overflow bin 6, bin interval is semi-open
    h(20.0); // put in overflow bin 6

    /*
      do a weighted fill using bh::weight, a wrapper for any type,
      which may appear at the beginning of the argument list
    */
    h(bh::weight(1.0), 0.1);

    /*
      iterate over bins with a fancy histogram iterator
      - order in which bins are iterated over is an implementation detail
      - iterator dereferences to histogram::element_type, which is defined by
        its storage class; by default something with value() and
        variance() methods; the first returns the
        actual count, the second returns a variance estimate of the count
        (see Rationale section for what this means)
      - idx(N) method returns the index of the N-th axis
      - bin(N_c) method returns current bin of N-th axis; the suffx _c turns
        the argument into a compile-time number, which is needed to return
        different `bin_type`s for different axes
      - `bin_type` usually is a semi-open interval representing the bin, whose
        edges can be accessed with methods `lower()` and `upper()`, but the
        implementation depends on the axis, please look it up in the reference
    */
    for (auto it = h.begin(); it != h.end(); ++it) {
      const auto bin = it.bin(0_c);
      std::cout << "bin " << it.idx(0) << " x in ["
                << bin.lower() << ", " << bin.upper() << "): "
                << it->value() << " +/- "
                << std::sqrt(it->variance())
                << std::endl;
    }

    /* program output: (note that under- and overflow bins appear at the end)

    bin 0 x in [-1.0, -0.5): 1 +/- 1
    bin 1 x in [-0.5,  0.0): 0 +/- 0
    bin 2 x in [ 0.0,  0.5): 1 +/- 1
    bin 3 x in [ 0.5,  1.0): 0 +/- 0
    bin 4 x in [ 1.0,  1.5): 1 +/- 1
    bin 5 x in [ 1.5,  2.0): 0 +/- 0
    bin 6 x in [ 2.0, inf): 2 +/- 1.41421
    bin -1 x in [-inf, -1): 1 +/- 1

    */
}

//]
