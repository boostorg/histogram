// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/python.hpp>
#include <boost/histogram.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

namespace br = boost::random;
namespace bh = boost::histogram;
namespace bp = boost::python;

void process(bh::histogram<bh::Dynamic, bh::axis::builtins>& h) {
  br::mt19937 gen;
  br::normal_distribution<> norm;
  // fill histogram
  for (int i = 0; i < 1000; ++i)
      h.fill(norm(gen), norm(gen));
}

BOOST_PYTHON_MODULE(cpp_filler) {
  bp::def("process", process);
}
