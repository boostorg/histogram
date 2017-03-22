// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "serialization_suite.hpp"
#include <boost/histogram/axis.hpp>
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/utility.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/histogram_ostream_operators.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/apply_visitor.hpp>
#ifdef HAVE_NUMPY
# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL boost_histogram_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# include <numpy/arrayobject.h>
#endif

#ifndef BOOST_HISTOGRAM_AXIS_LIMIT
  #define BOOST_HISTOGRAM_AXIS_LIMIT 32
#endif

namespace boost {
namespace histogram {

struct axis_visitor : public static_visitor<python::object>
{
  template <typename T>
  python::object operator()(const T& t) const { return python::object(t); }
};

python::object
histogram_axis(const dynamic_histogram<>& self, unsigned i)
{
  return apply_visitor(axis_visitor(), self.axis(i));
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

  const unsigned dim = len(args) - 1;

  // normal constructor
  dynamic_histogram<>::axes_type axes;
  for (unsigned i = 0; i < dim; ++i) {
    object pa = args[i + 1];
    extract<regular_axis<>> er(pa);
    if (er.check()) { axes.push_back(er()); continue; }
    extract<circular_axis<>> ep(pa);
    if (ep.check()) { axes.push_back(ep()); continue; }
    extract<variable_axis<>> ev(pa);
    if (ev.check()) { axes.push_back(ev()); continue; }
    extract<integer_axis> ei(pa);
    if (ei.check()) { axes.push_back(ei()); continue; }
    extract<category_axis> ec(pa);
    if (ec.check()) { axes.push_back(ec()); continue; }
    std::string msg = "require an axis object, got ";
    msg += extract<std::string>(pa.attr("__class__").attr("__name__"))();
    PyErr_SetString(PyExc_TypeError, msg.c_str());
    throw_error_already_set();
  }
  return pyinit(axes.begin(), axes.end());
}

python::object
histogram_fill(python::tuple args, python::dict kwargs) {
  using namespace python;

  const unsigned nargs = len(args);
  dynamic_histogram<>& self = extract<dynamic_histogram<>&>(args[0]);

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
            self.wfill(*w, v, v+self.dim());
          }

          Py_DECREF(aw);
        } else {
          PyErr_SetString(PyExc_ValueError, "w is not a sequence");
          throw_error_already_set();
        }
      } else {
        for (unsigned i = 0; i < dims[0]; ++i) {
          double* v = reinterpret_cast<double*>(PyArray_GETPTR1(a, i));
          self.fill(v, v+self.dim());
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

  if (dim > BOOST_HISTOGRAM_AXIS_LIMIT) {
    std::ostringstream os;
    os << "too many axes, maximum is " << BOOST_HISTOGRAM_AXIS_LIMIT;
    PyErr_SetString(PyExc_RuntimeError, os.str().c_str());
    throw_error_already_set();
  }

  double v[BOOST_HISTOGRAM_AXIS_LIMIT];
  for (unsigned i = 0; i < dim; ++i)
    v[i] = extract<double>(args[1 + i]);

  if (ow.is_none()) {
    self.fill(v, v+self.dim());
  } else {
    const double w = extract<double>(ow);
    self.wfill(w, v, v+self.dim());
  }

  return object();
}

python::object
histogram_value(python::tuple args, python::dict kwargs) {
  using namespace python;
  const dynamic_histogram<>& self = extract<const dynamic_histogram<>&>(args[0]);

  const unsigned dim = len(args) - 1;
  if (self.dim() != dim) {
    PyErr_SetString(PyExc_RuntimeError, "wrong number of arguments");
    throw_error_already_set();
  }

  if (dim > BOOST_HISTOGRAM_AXIS_LIMIT) {
    std::ostringstream os;
    os << "too many axes, maximum is " << BOOST_HISTOGRAM_AXIS_LIMIT;
    PyErr_SetString(PyExc_RuntimeError, os.str().c_str());
    throw_error_already_set();
  }

  if (kwargs) {
    PyErr_SetString(PyExc_RuntimeError, "no keyword arguments allowed");
    throw_error_already_set();
  }

  int idx[BOOST_HISTOGRAM_AXIS_LIMIT];
  for (unsigned i = 0; i < self.dim(); ++i)
    idx[i] = extract<int>(args[1 + i]);

  return object(self.value(idx + 0, idx + self.dim()));
}

python::object
histogram_variance(python::tuple args, python::dict kwargs) {
  using namespace python;
  const dynamic_histogram<>& self = extract<const dynamic_histogram<>&>(args[0]);

  const unsigned dim = len(args) - 1;
  if (self.dim() != dim) {
    PyErr_SetString(PyExc_RuntimeError, "wrong number of arguments");
    throw_error_already_set();
  }

  if (dim > BOOST_HISTOGRAM_AXIS_LIMIT) {
    std::ostringstream os;
    os << "too many axes, maximum is " << BOOST_HISTOGRAM_AXIS_LIMIT;
    PyErr_SetString(PyExc_RuntimeError, os.str().c_str());
    throw_error_already_set();
  }

  if (kwargs) {
    PyErr_SetString(PyExc_RuntimeError, "no keyword arguments allowed");
    throw_error_already_set();
  }

  int idx[BOOST_HISTOGRAM_AXIS_LIMIT];
  for (unsigned i = 0; i < self.dim(); ++i)
    idx[i] = extract<int>(args[1 + i]);

  return object(self.variance(idx + 0, idx + self.dim()));
}

std::string
histogram_repr(const dynamic_histogram<>& h) {
    std::ostringstream os;
    os << h;
    return os.str();
}

struct storage_access {
  static
  python::object
  array_interface(dynamic_histogram<>& self) {
    const auto& b = self.storage_.buffer_;
    if (b.type_.id_ == 5) {
      // PyErr_SetString(PyExc_KeyError, "cannot convert multiprecision storage to numpy array");
      // python::throw_error_already_set();
      // workaround: for some reason, exception is ignored here
      return python::object();
    }

    python::dict d;
    python::list shapes;
    python::list strides;
    std::size_t stride = 1;
    if (b.type_.id_ == 6) {
      stride *= sizeof(double);
      d["typestr"] = python::str("|f") + python::str(stride);
      strides.append(stride);
      stride *= 2;
      shapes.append(2);
    } else {
      stride *= b.type_.depth_;
      d["typestr"] = python::str("|u") + python::str(stride);
    }
    for (unsigned i = 0; i < self.dim(); ++i) {
      shapes.append(shape(self.axis(i)));
      strides.append(stride);
      stride *= shape(self.axis(i));
    }
    d["shape"] = python::tuple(shapes);
    d["data"] = python::make_tuple(reinterpret_cast<uintptr_t>(b.ptr_), false);
    d["strides"] = python::tuple(strides);
    return d;
  }
};

void register_histogram()
{
  using namespace python;
  using python::arg;
  docstring_options dopt(true, true, false);

  // used to pass arguments from raw python init to specialized C++ constructor
  class_<dynamic_histogram<>::axes_type::iterator>("axes_iterator", no_init);

  class_<dynamic_histogram<>, boost::shared_ptr<dynamic_histogram<>>>("histogram",
    "N-dimensional histogram for real-valued data.",
    no_init)
    .def("__init__", raw_function(histogram_init),
       ":param axis args: axis objects"
       "\nPass one or more axis objects to define"
       "\nthe dimensions of the dynamic_histogram<>.")
    // shadowed C++ ctors
    .def(init<dynamic_histogram<>::axes_type::iterator,
              dynamic_histogram<>::axes_type::iterator>())
    .add_property("__array_interface__",
        &storage_access::array_interface)
    .add_property("dim", &dynamic_histogram<>::dim,
       "dimensions of the histogram (number of axes)")
    .def("axis", histogram_axis,
       ":param int i: index of the axis\n"
       ":returns: axis object for axis i",
       (arg("self"), arg("i") = 0))
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
    .add_property("sum", &dynamic_histogram<>::sum)
    .def("value", raw_function(histogram_value),
       ":param int args: indices of the bin"
       "\n:return: count for the bin")
    .def("variance", raw_function(histogram_variance),
       ":param int args: indices of the bin"
       "\n:return: variance estimate for the bin")
    .def("__repr__", histogram_repr,
       ":returns: string representation of the histogram")
    .def(self == self)
    .def(self += self)
    .def_pickle(serialization_suite<dynamic_histogram<>>())
    ;
}

} // NS histogram
} // NS boost
