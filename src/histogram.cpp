// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "serialization_suite.hpp"
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#ifdef HAVE_NUMPY
# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL boost_histogram_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# include <numpy/arrayobject.h>
#endif

namespace boost {
namespace histogram {

constexpr unsigned boost_histogram_axis_limit = 10;
using dhistogram = histogram<>;
using axes_t = std::vector<axis_t>;

struct axis_visitor : public static_visitor<python::object>
{
  template <typename T>
  python::object operator()(const T& t) const { return python::object(T(t)); }
};

python::object
histogram_axis(const dhistogram& self, unsigned i)
{
  return apply_visitor(axis_visitor(), self.axis<axis_t>(i));
}

python::object
histogram_init(python::tuple args, python::dict kwargs) {
  using namespace python;
  using python::tuple;

  object self = args[0];
  object pyinit = self.attr("__init__");

  if (kwargs) {
    PyErr_SetString(PyExc_RuntimeError, "no keyword arguments allowed");
    throw_error_already_set();
  }

  // normal constructor
  axes_t axes;
  for (unsigned i = 1, n = len(args); i < n; ++i) {
    object pa = args[i];
    extract<regular_axis> er(pa);
    if (er.check()) { axes.push_back(er()); continue; }
    extract<polar_axis> ep(pa);
    if (ep.check()) { axes.push_back(ep()); continue; }
    extract<variable_axis> ev(pa);
    if (ev.check()) { axes.push_back(ev()); continue; }
    extract<category_axis> ec(pa);
    if (ec.check()) { axes.push_back(ec()); continue; }
    extract<integer_axis> ei(pa);
    if (ei.check()) { axes.push_back(ei()); continue; }
    std::string msg = "require an axis object, got ";
    msg += extract<std::string>(pa.attr("__class__").attr("__name__"))();
    PyErr_SetString(PyExc_TypeError, msg.c_str());
    throw_error_already_set();
  }
  return pyinit(axes);
}

python::object
histogram_fill(python::tuple args, python::dict kwargs) {
  using namespace python;

  const unsigned nargs = len(args);
  dhistogram& self = extract<dhistogram&>(args[0]);

  object ow;
  if (kwargs) {
    if (len(kwargs) > 1 || !kwargs.has_key("w")) {
      PyErr_SetString(PyExc_RuntimeError, "only keyword w allowed");
      throw_error_already_set();
    }
    ow = kwargs.get("w");
  }

#ifdef HAVE_NUMPY
  if (nargs == 2) {
    object o = args[1];
    if (PySequence_Check(o.ptr())) {
      PyArrayObject* a = reinterpret_cast<PyArrayObject*>
        (PyArray_FROM_OTF(o.ptr(), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
      if (!a) {
        PyErr_SetString(PyExc_ValueError, "could not convert sequence into array");
        throw_error_already_set();
      }

      npy_intp* dims = PyArray_DIMS(a);
      switch (PyArray_NDIM(a)) {
        case 1: 
        if (self.dim() > 1) {
          PyErr_SetString(PyExc_ValueError, "array has to be two-dimensional");
          throw_error_already_set();
        }
        break;
        case 2:
        if (self.dim() != dims[1])
        {
          PyErr_SetString(PyExc_ValueError, "size of second dimension does not match");
          throw_error_already_set();
        }
        break;
        default:
          PyErr_SetString(PyExc_ValueError, "array has wrong dimension");
          throw_error_already_set();
      }

      if (!ow.is_none()) {
        if (PySequence_Check(ow.ptr())) {
          PyArrayObject* aw = reinterpret_cast<PyArrayObject*>
            (PyArray_FROM_OTF(ow.ptr(), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
          if (!aw) {
            PyErr_SetString(PyExc_ValueError, "could not convert sequence into array");
            throw_error_already_set();            
          }

          if (PyArray_NDIM(aw) != 1) {
            PyErr_SetString(PyExc_ValueError, "array has to be one-dimensional");
            throw_error_already_set();
          }

          if (PyArray_DIMS(aw)[0] != dims[0]) {
            PyErr_SetString(PyExc_ValueError, "sizes do not match");
            throw_error_already_set();
          }

          for (unsigned i = 0; i < dims[0]; ++i) {
            double* v = reinterpret_cast<double*>(PyArray_GETPTR1(a, i) );
            double* w = reinterpret_cast<double*>(PyArray_GETPTR1(aw, i));
            self.wfill_iter(v, v+self.dim(), *w);
          }

          Py_DECREF(aw);
        } else {
          PyErr_SetString(PyExc_ValueError, "w is not a sequence");
          throw_error_already_set();
        }
      } else {
        for (unsigned i = 0; i < dims[0]; ++i) {
          double* v = reinterpret_cast<double*>(PyArray_GETPTR1(a, i));
          self.fill_iter(v, v+self.dim());
        }
      }

      Py_DECREF(a);
      return object();
    }
  }
#endif

  const unsigned dim = nargs - 1;
  if (dim != self.dim()) {
    PyErr_SetString(PyExc_RuntimeError, "wrong number of arguments");
    throw_error_already_set();      
  }

  double v[boost_histogram_axis_limit];
  for (unsigned i = 0; i < dim; ++i)
    v[i] = extract<double>(args[1 + i]);

  if (ow.is_none()) {
    self.fill_iter(v, v+self.dim());

  } else {
    const double w = extract<double>(ow);
    self.wfill_iter(v, v+self.dim(), w);
  }

  return object();
}

python::object
histogram_value(python::tuple args, python::dict kwargs) {
  using namespace python;
  const dhistogram& self = extract<const dhistogram&>(args[0]);

  if (self.dim() != (len(args) - 1)) {
    PyErr_SetString(PyExc_RuntimeError, "wrong number of arguments");
    throw_error_already_set();      
  }

  if (kwargs) {
    PyErr_SetString(PyExc_ValueError, "no keyword arguments allowed");
    throw_error_already_set();    
  }

  int idx[boost_histogram_axis_limit];
  for (unsigned i = 0; i < self.dim(); ++i)
    idx[i] = extract<int>(args[1 + i]);

  return object(self.value_iter(idx + 0, idx + self.dim()));
}

python::object
histogram_variance(python::tuple args, python::dict kwargs) {
  using namespace python;
  const dhistogram& self = extract<const dhistogram&>(args[0]);

  if (self.dim() != (len(args) - 1)) {
    PyErr_SetString(PyExc_RuntimeError, "wrong number of arguments");
    throw_error_already_set();      
  }

  if (kwargs) {
    PyErr_SetString(PyExc_RuntimeError, "no keyword arguments allowed");
    throw_error_already_set();    
  }

  int idx[boost_histogram_axis_limit];
  for (unsigned i = 0; i < self.dim(); ++i)
    idx[i] = extract<int>(args[1 + i]);

  return object(self.variance_iter(idx + 0, idx + self.dim()));
}

class histogram_access {
public:
  static
  python::dict
  histogram_array_interface(dhistogram& self) {
    python::dict d;
    python::list shape;
    for (unsigned i = 0; i < self.dim(); ++i)
      shape.append(self.shape(i));
    if (self.depth() == sizeof(detail::wtype)) {
      shape.append(2);
      d["typestr"] = python::str("<f") + python::str(sizeof(double));
    } else {
      d["typestr"] = python::str("<u") + python::str(self.depth());
    }
    d["shape"] = python::tuple(shape);
    d["data"] = python::make_tuple(reinterpret_cast<uintptr_t>(self.data()), false);
    return d;
  }
};

template <class Archive>
inline void serialize(Archive& ar, dhistogram& h, unsigned) {
  ar & boost::serialization::base_object<dhistogram::base_t>(h);
}

template <class Archive>
inline void serialize(Archive& ar, dhistogram::base_t& h, unsigned) {
  ar & h.axes_;
  ar & h.storage_;
}

void register_histogram()
{
  using namespace python;
  using python::arg;
  docstring_options dopt(true, true, false);

  // used to pass arguments from raw python init to specialized C++ constructor
  class_<axes_t>("axes_t", no_init);
  class_<axes_t::const_iterator>("axes_const_iterator", no_init);

  class_<dhistogram, boost::shared_ptr<dhistogram>>("histogram",
    "N-dimensional histogram for real-valued data.",
    no_init)
    .def("__init__", raw_function(histogram_init),
       ":param axis args: axis objects"
       "\nPass one or more axis objects to define"
       "\nthe dimensions of the histogram.")
    // shadowed C++ ctors
    .def(init<axes_t>())
    .add_property("__array_interface__",
            &histogram_access::histogram_array_interface)
    .add_property("dim", &dhistogram::dim,
            "dimensions of the histogram")
    .def("shape", &dhistogram::shape,
       ":param int i: index of the axis\n"
       ":returns: number of count fields for axis i\n"
       "  (bins + 2 if underflow and overflow"
       " bins are enabled, otherwise equal to bins",
       args("self", "i"))
    .def("axis", histogram_axis,
       ":param int i: index of the axis\n"
       ":returns: axis object for axis i",
       args("self", "i"))
    .def("fill", raw_function(histogram_fill),
       "Pass a sequence of values with a length n is"
       "\nequal to the dimensions of the histogram,"
       "\nand optionally a weight w for this fill"
       "\n(*int* or *float*)."
       "\n"
       "\nIf Numpy support is enabled, values may also"
       "\nbe a 2d-array of shape (m, n), where m is"
       "\nthe number of tuples, and optionally"
       "\nanother a second 1d-array w of shape (n,).")
    .add_property("depth", &dhistogram::depth)
    .add_property("sum", &dhistogram::sum)
    .def("value", raw_function(histogram_value),
       ":param int args: indices of the bin"
       "\n:return: count for the bin")
    .def("variance", raw_function(histogram_variance),
       ":param int args: indices of the bin"
       "\n:return: variance estimate for the bin")
    .def(self == self)
    .def(self += self)
    .def(self + self)
    .def_pickle(serialization_suite<dhistogram>())
    ;
}

}
}
