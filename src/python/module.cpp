#include <boost/python/module.hpp>
#ifdef HAVE_NUMPY
# define PY_ARRAY_UNIQUE_SYMBOL boost_histogram_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# include <numpy/arrayobject.h>
#endif

namespace boost {
namespace histogram {
  void register_axis_types();
  void register_basic_histogram();
  void register_histogram();
}
}

BOOST_PYTHON_MODULE(histogram)
{
#ifdef HAVE_NUMPY
  import_array();
#endif
  boost::histogram::register_axis_types();
  boost::histogram::register_basic_histogram();
  boost::histogram::register_histogram();
}
