// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/python.hpp>
#include <boost/python/def_visitor.hpp>
#include <boost/python/raw_function.hpp>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

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

  std::vector<std::string> c;
  for (int i = 1, n = len(args); i < n; ++i)
    c.push_back(extract<std::string>(args[i]));

  return self.attr("__init__")(axis::category(c.begin(), c.end(), label));
}

template <typename T> int axis_len(const T &t) {
  return t.bins() + int(std::is_floating_point<typename T::value_type>::value);
}

template <typename T> python::object axis_getitem(const T &t, int i) {
  if (i == axis_len(t)) {
    PyErr_SetString(PyExc_StopIteration, "no more");
    python::throw_error_already_set();
  }
  return python::object(t[i]);
}

template <> python::object axis_getitem(const axis::category &a, int i) {
  if (i == axis_len(a)) {
    PyErr_SetString(PyExc_StopIteration, "no more");
    python::throw_error_already_set();
  }
  return python::object(a[i].data());
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

template <class T>
struct axis_suite : public python::def_visitor<axis_suite<T>> {
  template <class Class> static void visit(Class &cl) {
    cl.add_property("bins", &T::bins, "Number of bins.");
    cl.add_property(
        "shape", &T::shape,
        "Number of bins, including possible over- and underflow bins.");
    cl.add_property(
        "label", axis_get_label<T>, axis_set_label<T>,
        "Name or description for the axis.");
    cl.def("index", &T::index, ":param float x: value"
                               "\n:returns: bin index for the passed value",
           python::args("self", "x"));
    cl.def("__len__", axis_len<T>, ":returns: number of bins for this axis",
           python::arg("self"));
    cl.def("__getitem__", axis_getitem<T>,
           is_same<T, axis::integer>::value
               ? ":returns: integer mapped to passed bin index"
               : is_same<T, axis::category>::value
                     ? ":returns: category mapped to passed bin index"
                     : ":returns: low edge of the bin",
           python::args("self", "index"));
    cl.def("__repr__", axis_repr<T>,
           ":returns: string representation of this axis", python::arg("self"));
    cl.def(python::self == python::self);
  }
};

} // namespace

void register_axis_types() {
  using namespace python;
  using python::arg;
  docstring_options dopt(true, true, false);

  class_<axis::regular<>>("regular",
                         "An axis for real-valued data and bins of equal width."
                         "\nBinning is a O(1) operation.",
                         no_init)
      .def(init<unsigned, double, double, const std::string &, bool>(
          (arg("self"), arg("bin"), arg("min"), arg("max"),
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
      "\nthe problem domain allows it, prefer a regular.",
      no_init)
      .def("__init__", raw_function(variable_init))
      .def(init<const axis::variable<> &>())
      .def(axis_suite<axis::variable<>>());

  class_<axis::integer>("integer",
                       "An axis for a contiguous range of integers."
                       "\nThere are no underflow/overflow bins for this axis."
                       "\nBinning is a O(1) operation.",
                       no_init)
      .def(init<int, int, const std::string &, bool>(
          (arg("self"), arg("min"), arg("max"), arg("label") = std::string(),
           arg("uoflow") = true)))
      .def(axis_suite<axis::integer>());

  class_<axis::category>("category",
                        "An axis for enumerated categories. The axis stores the"
                        "\ncategory labels, and expects that they are addressed"
                        "\nusing an integer from 0 to n-1. There are no"
                        "\nunderflow/overflow bins for this axis."
                        "\nBinning is a O(1) operation.",
                        no_init)
      .def("__init__", raw_function(category_init))
      .def(init<const axis::category &>())
      .def(axis_suite<axis::category>());
}
}
}
