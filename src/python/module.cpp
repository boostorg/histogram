#include <boost/python/module.hpp>
#ifdef USE_NUMPY
# define PY_ARRAY_UNIQUE_SYMBOL boost_histogram_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# include <numpy/arrayobject.h>
#endif

namespace boost {
namespace histogram {
  void register_histogram_base();
  void register_histogram();
}
}

BOOST_PYTHON_MODULE(histogram)
{
#ifdef USE_NUMPY
  import_array();
#endif
  boost::histogram::register_histogram_base();
  boost::histogram::register_histogram();
}
