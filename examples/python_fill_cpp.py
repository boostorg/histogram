import histogram as bh
import cpp_filler

h = bh.histogram(bh.axis.regular(5, -5, 5, "x"),
                 bh.axis.regular(5, -5, 5, "y"))

cpp_filler.process(h) # histogram is filled with input values in c++

for iy in range(h.axis(1).bins):
    for ix in range(h.axis(0).bins):
        print "%3i" % h.value(ix, iy),
    print