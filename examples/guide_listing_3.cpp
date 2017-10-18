#include <boost/histogram.hpp>

namespace bh = boost::histogram;

int main() {
	// create a 1d-histogram for dice throws, eyes are always between 1 and 6
	auto h = bh::make_static_histogram(bh::axis::integer(1, 6, "eyes", false));
	// do something with h
}
