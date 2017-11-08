// also see examples/example_2d.cpp
#include <boost/histogram.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

namespace br = boost::random;
namespace bh = boost::histogram;

int main() {
	br::mt19937 gen;
	br::normal_distribution<> norm;
	auto h = bh::make_static_histogram(
		bh::axis::regular<>(100, -5, 5, "x"),
		bh::axis::regular<>(100, -5, 5, "y")
	);
	for (int i = 0; i < 1000; ++i)
		h.fill(norm(gen), norm(gen));
	// h is now filled
}
