// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/python/module.hpp>
#ifdef HAVE_NUMPY
# define PY_ARRAY_UNIQUE_SYMBOL boost_histogram_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# include <numpy/arrayobject.h>
# if PY_MAJOR_VERSION >= 3
static void* init_numpy() { import_array(); return NULL; }
# else
static void init_numpy() { import_array(); }
# endif
#endif

namespace boost {
namespace histogram {
  void register_axis_types();
  void register_histogram();
}
}

BOOST_PYTHON_MODULE(histogram)
{
#ifdef HAVE_NUMPY
  init_numpy();
#endif
  boost::histogram::register_axis_types();
  boost::histogram::register_histogram();
}
