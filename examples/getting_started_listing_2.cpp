#include <boost/histogram.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cstdlib>

namespace br = boost::random;
namespace bh = boost::histogram;

int main() {
    /*
        create dynamic histogram using `make_dynamic_histogram`
        - axis can be passed directly, just like for `make_static_histogram`
        - in addition, also accepts iterators over a sequence of axes
    */
    std::vector<bh::axis::any<>> axes = {bh::axis::regular<>(5, -5, 5, "x"),
                                         bh::axis::regular<>(5, -5, 5, "y")};
    auto h = bh::make_dynamic_histogram(axes.begin(), axes.end());

    // fill histogram, random numbers are generated on the fly
    br::mt19937 gen;
    br::normal_distribution<> norm;
    for (int i = 0; i < 1000; ++i)
        h.fill(norm(gen), norm(gen));

    /*
        print histogram
        - in opposition to the static case, we need to cast axis to correct
          type since dynamic histogram cannot do this automatically
        -
    */
    for (const auto& ybin : h.axis(1)) {
        for (const auto& xbin : h.axis(0)) {
            std::printf("%3.0f ", h.value(xbin.first, ybin.first));
        }
        std::printf("\n");
    }
}
