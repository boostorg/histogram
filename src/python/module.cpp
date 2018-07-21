// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/scope.hpp>
#ifdef HAVE_NUMPY
#include <boost/python/numpy.hpp>
#endif

void register_axis_types();
void register_histogram();

BOOST_PYTHON_MODULE(histogram) {
  using namespace boost::python;
  scope current;
#ifdef HAVE_NUMPY
  numpy::initialize();
  current.attr("HAVE_NUMPY") = true;
#else
  current.attr("HAVE_NUMPY") = false;
#endif
  object axis_module = object(borrowed(PyImport_AddModule("histogram.axis")));
  current.attr("axis") = axis_module;
  {
    scope current = axis_module;
    register_axis_types();
  }
  register_histogram();
}
