# also see examples/create_python_fill_cpp.py and examples/module_cpp_filler.cpp
import histogram as bh
import cpp_filler

h = bh.histogram(bh.axis.regular(100, -1, 1),
                 bh.axis.integer(0, 10))

cpp_filler.process(h) # histogram is filled with input values

# continue with statistical analysis of h
