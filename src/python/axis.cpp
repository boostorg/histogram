#include <boost/histogram/axis.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/def_visitor.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <sstream>
#include <string>

namespace boost {
namespace histogram {

namespace {

python::object
variable_axis_init(python::tuple args, python::dict kwargs) {
    using namespace python;
    using python::tuple;

    object self = args[0];
    object pyinit = self.attr("__init__");

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
        tuple kv = kwargs.popitem();
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
    return pyinit(v, label, uoflow);
}

python::object
category_axis_init(python::tuple args, python::dict kwargs) {
    using namespace python;

    object self = args[0];
    object pyinit = self.attr("__init__");

    if (len(args) == 1) {
        PyErr_SetString(PyExc_TypeError, "require at least one argument");
        throw_error_already_set();
    }

    if (len(kwargs) > 0) {
        PyErr_SetString(PyExc_TypeError, "unknown keyword argument");
        throw_error_already_set();        
    }

    if (len(args) == 2) {
        extract<std::string> es(args[1]);
        if (es.check())
            pyinit(es);
        else {
            PyErr_SetString(PyExc_TypeError, "require one or several string arguments");
            throw_error_already_set();
        }
    }

    std::vector<std::string> c;
    for (int i = 1, n = len(args); i < n; ++i)
        c.push_back(extract<std::string>(args[i]));

    return pyinit(c);
}

template <typename T>
int
axis_len(const T& t) {
    return t.bins() + 1;
}

template <>
int
axis_len(const category_axis& t) {
    return t.bins();
}

template <>
int
axis_len(const integer_axis& t) {
    return t.bins();
}

template <typename T>
typename T::value_type
axis_getitem(const T& t, int i) {
    if (i == axis_len(t)) {
        PyErr_SetString(PyExc_StopIteration, "no more");
        python::throw_error_already_set();
    }
    return t[i];
}

template <typename T>
std::string
axis_repr(const T& t) {
    std::ostringstream os;
    os << t;
    return os.str();
}

template<typename T>
struct has_index_method
{
    struct yes { char x[1]; };
    struct no { char x[2]; };
    template<typename U, int (U::*)(double) const> struct SFINAE {};
    template<typename U> static yes test( SFINAE<U, &U::index>* );
    template<typename U> static no  test( ... );
    enum { value = sizeof(test<T>(0)) == sizeof(yes) };
};

template <class T>
struct axis_suite : public python::def_visitor<axis_suite<T> > {    

    template <typename Class, typename U>
    static
    typename enable_if_c<has_index_method<U>::value, void>::type
    add_axis_index(Class& cl) {
        cl.def("index", &U::index,
               ":param float x: value"
               "\n:returns: bin index for the passed value",
               python::args("self", "x"));
    }

    template <typename Class, typename U>
    static
    typename disable_if_c<has_index_method<U>::value, void>::type
    add_axis_index(Class& cl) {}

    template <typename Class, typename U>
    static
    typename enable_if<is_base_of<axis_base, U>, void>::type
    add_axis_label(Class& cl) {
        cl.add_property("label",
                        python::make_function((const std::string&(U::*)() const) &U::label,
                                              python::return_value_policy<python::copy_const_reference>()),
                        (void(U::*)(const std::string&)) &U::label,
                        "Name or description for the axis.");
    }

    template <typename Class, typename U>
    static
    typename disable_if<is_base_of<axis_base, U>, void>::type
    add_axis_label(Class& cl) {}

    template <class Class>
    static void
    visit(Class& cl)
    {
        cl.add_property("bins", &T::bins);
        add_axis_index<Class, T>(cl);
        add_axis_label<Class, T>(cl);
        cl.def("__len__", axis_len<T>,
               ":returns: number of bins for this axis",
               python::arg("self"));
        cl.def("__getitem__", axis_getitem<T>,
               is_same<T, integer_axis>::value ?
                 ":returns: integer mapped to passed bin index" :
                 is_same<T, category_axis>::value ?
                   ":returns: category mapped to passed bin index" :
                   ":returns: low edge of the bin",
               python::args("self", "index"));
        cl.def("__repr__", axis_repr<T>,
               ":returns: string representation of this axis",
               python::arg("self"));
        cl.def(python::self == python::self);
    }
};

} // namespace

void register_axis_types()
{
  using namespace python;
  using python::arg;
  docstring_options dopt(true, true, false);

  // used to pass arguments from raw python init to specialized C++ constructors
  class_<std::vector<double> >("vector_double", no_init);
  class_<std::vector<std::string> >("vector_string", no_init);

  class_<regular_axis>("regular_axis",
    "An axis for real-valued data and bins of equal width."
    "\nBinning is a O(1) operation.",
    no_init)
    .def(init<unsigned, double, double, std::string, bool>(
         (arg("self"), arg("bin"), arg("min"), arg("max"),
          arg("label") = std::string(),
          arg("uoflow") = true)))
    .def(axis_suite<regular_axis>())
    ;

  class_<polar_axis>("polar_axis",
    "An axis for real-valued angles."
    "\nThere are no overflow/underflow bins for this axis,"
    "\nsince the axis is circular and wraps around after 2pi."
    "\nBinning is a O(1) operation.",
    no_init)
    .def(init<unsigned, double, std::string>(
         (arg("self"), arg("bin"), arg("start") = 0.0,
          arg("label") = std::string())))
    .def(axis_suite<polar_axis>())
    ;

  class_<variable_axis>("variable_axis",
    "An axis for real-valued data and bins of varying width."
    "\nBinning is a O(log(N)) operation. If speed matters and"
    "\nthe problem domain allows it, prefer a regular_axis.",
    no_init)
    .def("__init__", raw_function(variable_axis_init))
    .def(init<std::vector<double>, std::string, bool>())
    .def(axis_suite<variable_axis>())
    ;

  class_<category_axis>("category_axis",
    "An axis for enumerated categories. The axis stores the"
    "\ncategory labels, and expects that they are addressed"
    "\nusing an integer from 0 to n-1. There are no"
    "\nunderflow/overflow bins for this axis."
    "\nBinning is a O(1) operation.",
    no_init)
    .def("__init__", raw_function(category_axis_init))
    .def(init<std::string>())
    .def(init<std::vector<std::string> >())
    .def(axis_suite<category_axis>())
    ;

  class_<integer_axis>("integer_axis",
    "An axis for a contiguous range of integers."
    "\nThere are no underflow/overflow bins for this axis."
    "\nBinning is a O(1) operation.",
    no_init)
    .def(init<int, int, std::string, bool>(
         (arg("self"), arg("min"), arg("max"),
          arg("label") = std::string(),
          arg("uoflow") = true)))
    .def(axis_suite<integer_axis>())
    ;
}

}
}
