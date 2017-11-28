// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/python/module.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/object.hpp>
#ifdef HAVE_NUMPY
#include <boost/python/numpy.hpp>
#endif

namespace boost {
namespace histogram {
void register_axis_types();
void register_histogram();
}
}

BOOST_PYTHON_MODULE(histogram) {
  using namespace boost::python;
#ifdef HAVE_NUMPY
  numpy::initialize();
#endif
  scope current;
  object axis_module = object(
    borrowed(PyImport_AddModule("histogram.axis"))
  );
  current.attr("axis") = axis_module;
  {
    scope current = axis_module;
    boost::histogram::register_axis_types();
  }
  boost::histogram::register_histogram();
}
