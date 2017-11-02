// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utility.hpp"
#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/python.hpp>
#include <boost/python/def_visitor.hpp>
#include <boost/python/raw_function.hpp>
#ifdef HAVE_NUMPY
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL boost_histogram_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

namespace boost {
namespace histogram {

namespace {

python::object variable_init(python::tuple args, python::dict kwargs) {
  using namespace python;

  object self = args[0];

  if (len(args) < 2) {
    PyErr_SetString(PyExc_TypeError, "require at least two arguments");
    throw_error_already_set();
  }

  std::vector<double> v;
  for (int i = 1, n = len(args); i < n; ++i) {
    v.push_back(extract<double>(args[i]));
  }

  std::string label;
  bool uoflow = true;
  while (len(kwargs) > 0) {
    python::tuple kv = kwargs.popitem();
    std::string k = extract<std::string>(kv[0]);
    object v = kv[1];
    if (k == "label")
      label = extract<std::string>(v);
    else if (k == "uoflow")
      uoflow = extract<bool>(v);
    else {
      std::stringstream s;
      s << "keyword " << k << " not recognized";
      PyErr_SetString(PyExc_KeyError, s.str().c_str());
      throw_error_already_set();
    }
  }

  return self.attr("__init__")(
      axis::variable<>(v.begin(), v.end(), label, uoflow));
}

python::object category_init(python::tuple args, python::dict kwargs) {
  using namespace python;

  object self = args[0];

  if (len(args) == 1) {
    PyErr_SetString(PyExc_TypeError, "require at least one argument");
    throw_error_already_set();
  }

  std::string label;
  while (len(kwargs) > 0) {
    python::tuple kv = kwargs.popitem();
    std::string k = extract<std::string>(kv[0]);
    object v = kv[1];
    if (k == "label")
      label = extract<std::string>(v);
    else {
      std::stringstream s;
      s << "keyword " << k << " not recognized";
      PyErr_SetString(PyExc_KeyError, s.str().c_str());
      throw_error_already_set();
    }
  }

  std::vector<int> c;
  for (int i = 1, n = len(args); i < n; ++i)
    c.push_back(extract<int>(args[i]));

  return self.attr("__init__")(axis::category<>(c.begin(), c.end(), label));
}

template <typename A> python::object axis_getitem(const A &a, int i) {
  if (i == a.size()) {
    PyErr_SetString(PyExc_StopIteration, "no more");
    python::throw_error_already_set();
  }
  return python::object(a[i]);
}

template <> python::object axis_getitem(const axis::category<> &a, int i) {
  if (i == a.size()) {
    PyErr_SetString(PyExc_StopIteration, "no more");
    python::throw_error_already_set();
  }
  return python::object(a[i]);
}

template <typename T> std::string axis_repr(const T &t) {
  std::ostringstream os;
  os << t;
  return os.str();
}

template <typename T> void axis_set_label(T& t, python::str s) {
  const char* d = python::extract<const char*>(s);
  const auto n = python::len(s);
  t.label(string_view(d, n));
}

template <typename T> python::str axis_get_label(const T& t) {
  auto s = t.label();
  return {s.data(), s.size()};
}

#ifdef HAVE_NUMPY
template <typename Axis> python::object axis_array_interface(const Axis& axis) {
  python::dict d;
  auto shape = python::make_tuple(axis.size()+1);
  d["shape"] = shape;
  // d["typestr"] = dtype_typestr<typename Axis::bin_type>();
  d["typestr"] = "|f8";
  // make new array, and pass it to Python
  auto dim = 1;
  npy_intp shapes2[1] = { axis.size()+1 };
  auto *a = (PyArrayObject*)PyArray_SimpleNew(dim, shapes2, NPY_DOUBLE);
  auto *buf = (double *)PyArray_DATA(a);
  PyArray_CLEARFLAGS(a, NPY_ARRAY_WRITEABLE);
  // auto a = python::numpy::empty(shape, python::numpy::dtype::get_builtin<typename Axis::bin_type>());
  // auto buf = reinterpret_cast<typename Axis::bin_type*>(axis.get_data());
  for (auto i = 0; i < axis.size()+1; ++i)
    buf[i] = axis[i].lower();
  d["data"] = python::object(python::handle<>((PyObject*)a));
  d["version"] = 3;
  return d;
}

template <> python::object axis_array_interface<axis::category<>>(const axis::category<>& axis) {
  python::dict d;
  auto shape = python::make_tuple(axis.size());
  d["shape"] = shape;
  d["typestr"] = python::dtype_typestr<int>();
  // make new array, and pass it to Python
  auto dim = 1;
  npy_intp shapes2[1] = { axis.size() };
  auto *a = (PyArrayObject*)PyArray_SimpleNew(dim, shapes2, NPY_INT);
  auto *buf = (int *)PyArray_DATA(a);
  PyArray_CLEARFLAGS(a, NPY_ARRAY_WRITEABLE);
  // auto a = python::numpy::empty(shape, python::numpy::dtype::get_builtin<typename Axis::bin_type>());
  // auto buf = reinterpret_cast<typename Axis::bin_type*>(axis.get_data());
  for (auto i = 0; i < axis.size(); ++i)
    buf[i] = axis[i];
  d["data"] = python::object(python::handle<>((PyObject*)a));
  d["version"] = 3;
  return d;
}
#endif

template <class T>
struct axis_suite : public python::def_visitor<axis_suite<T>> {
  template <class Class> static void visit(Class &cl) {
    cl.add_property(
        "shape", &T::shape,
        "Number of bins, including over-/underflow bins if they are present.");
    cl.add_property(
        "label", axis_get_label<T>, axis_set_label<T>,
        "Name or description for the axis.");
    cl.def("index", &T::index, ":param float x: value"
                               "\n:returns: bin index for the passed value",
           python::args("self", "x"));
    cl.def("__len__", &T::size,
           ":returns: number of bins, excluding over-/underflow bins.",
           python::arg("self"));
    cl.def("__getitem__", axis_getitem<T>,
           ":param integer i: bin index"
           "\n:returns: bin corresponding to index",
           python::args("self", "i"));
    cl.def("__repr__", axis_repr<T>,
           ":returns: string representation of this axis", python::arg("self"));
    cl.def(python::self == python::self);
#ifdef HAVE_NUMPY
    cl.add_property("__array_interface__", &axis_array_interface<T>);
#endif
  }
};

} // namespace

void register_axis_types() {
  using namespace python;
  using python::arg;
  docstring_options dopt(true, true, false);

  class_<interval<double>>(
      "interval_double",
      no_init)
      .add_property("lower",
                    make_function(&interval<double>::lower,
                                  return_value_policy<return_by_value>()))
      .add_property("upper",
                    make_function(&interval<double>::upper,
                                  return_value_policy<return_by_value>()))
      ;

  class_<interval<int>>(
      "interval_int",
      no_init)
      .add_property("lower",
                    make_function(&interval<int>::lower,
                                  return_value_policy<return_by_value>()))
      .add_property("upper",
                    make_function(&interval<int>::upper,
                                  return_value_policy<return_by_value>()))
      ;

  class_<axis::regular<>>(
      "regular",
      "An axis for real-valued data and bins of equal width."
      "\nBinning is a O(1) operation.",
      no_init)
      .def(init<unsigned, double, double, const std::string &, bool>(
          (arg("self"), arg("bin"), arg("lower"), arg("upper"),
           arg("label") = std::string(), arg("uoflow") = true)))
      .def(axis_suite<axis::regular<>>());

  class_<axis::circular<>>(
      "circular",
      "An axis for real-valued angles."
      "\nThere are no overflow/underflow bins for this axis,"
      "\nsince the axis is circular and wraps around after reaching"
      "\nthe perimeter value. Binning is a O(1) operation.",
      no_init)
      .def(init<unsigned, double, double, const std::string &>(
          (arg("self"), arg("bin"), arg("phase") = 0.0,
           arg("perimeter") = math::double_constants::two_pi,
           arg("label") = std::string())))
      .def(axis_suite<axis::circular<>>());

  class_<axis::variable<>>(
      "variable",
      "An axis for real-valued data and bins of varying width."
      "\nBinning is a O(log(N)) operation. If speed matters and"
      "\nthe problem domain allows it, prefer a regular axis.",
      no_init)
      .def("__init__", raw_function(variable_init))
      .def(init<const axis::variable<> &>())
      .def(axis_suite<axis::variable<>>());

  class_<axis::integer<>>(
      "integer",
      "An axis for a contiguous range of integers with bins"
      "\nthat are one integer wide. Faster than a regular axis."
      "\nBinning is a O(1) operation.",
      no_init)
      .def(init<int, int, const std::string &, bool>(
          (arg("self"), arg("lower"), arg("upper"), arg("label") = std::string(),
           arg("uoflow") = true)))
      .def(axis_suite<axis::integer<>>());

  class_<axis::category<>>(
      "category",
      "An axis for set of unique integer values. Each value is mapped to"
      "\na corresponding bin, following the order of the arguments in"
      "\nthe constructor."
      "\nBinning is a O(1) operation.",
      no_init)
      .def("__init__", raw_function(category_init))
      .def(init<const axis::category<> &>())
      .def(axis_suite<axis::category<>>());
}
}
}
