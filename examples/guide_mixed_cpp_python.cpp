//[ guide_mixed_cpp_python_part_cpp

#include <boost/python.hpp>
#include <boost/histogram.hpp>

namespace bh = boost::histogram;
namespace bp = boost::python;

// function that runs in C++ and accepts reference to dynamic histogram
void process(bh::dynamic_histogram<>& h) {
  // fill histogram, in reality this would be arbitrarily complex code
  for (int i = 0; i < 4; ++i)
      h(0.25 * i, i);
}

// a minimal Python module, which exposes the process function to Python
BOOST_PYTHON_MODULE(cpp_filler) {
  bp::def("process", process);
}

//]
