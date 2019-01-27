import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import re
import sys

bench = defaultdict(lambda:[])
for iline, line in enumerate(open(sys.argv[1])):
    if iline < 3:
        continue
    # Naive/(tuple, 3, inner)/4    3.44 ns
    m = re.match("(\S+)/\((\S+), (\d), (\S+)\)/(\d+)\s*([0-9\.]+) ns", line)
    name = m.group(1)
    hist = m.group(2)
    dim = int(m.group(3))
    cov = m.group(4)
    nbins = int(m.group(5))
    time = float(m.group(6))
    bench[(name, hist, dim, cov)].append((int(nbins) ** dim, time))

fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.18)
for iaxis, axis_type in enumerate(("tuple", "vector", "vector_of_variant")):
    plt.sca(ax[iaxis])
    plt.title(axis_type, y=1.02)
    handles = []
    for (name, axis_t, dim, cov), v in bench.items():
        if axis_t != axis_type: continue
        if cov != "inner": continue
        v = np.sort(v, axis=0).T
        # if "semi_dynamic" in axis: continue
        name2, col, ls = {
            "Naive": ("nested for", "r", "--"),
            "Indexed": ("indexed", "b", ":")}.get(name, (name, "k", "-"))
        h = plt.plot(v[0], v[1] / v[0], color=col, ls=ls, lw=dim,
                     label=r"%s: $D=%i$" % (name2, dim))[0]
        handles.append(h)
        handles.sort(key=lambda x: x.get_label())
    plt.loglog()
    plt.legend(handles=handles, fontsize="xx-small")
    plt.ylabel("CPU time in ns per bin")
    # plt.savefig("iteration_performance.svg")
plt.figtext(0.5, 0.05, "number of bins", ha="center")
plt.show()
