#include <boost/histogram.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

int main() {
    using namespace boost;
    random::mt19937 gen;
    random::normal_distribution<> norm;
    auto h = histogram::make_static_histogram(
        histogram::regular_axis<>(100, -5, 5, "x"),
        histogram::regular_axis<>(100, -5, 5, "y")
    );
    for (int i = 0; i < 1000; ++i)
        h.fill(norm(gen), norm(gen));
    // h is now filled
}
