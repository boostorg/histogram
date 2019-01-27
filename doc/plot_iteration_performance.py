import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import re
import sys

data = json.load(open(sys.argv[1]))

bench = defaultdict(lambda:[])
for x in data["benchmarks"]:
    if x["run_type"] != "aggregate":
        continue
    if x["aggregate_name"] != "mean":
        continue
    name, args, nbins = x["run_name"].split("/")
    # "run_name": "Naive/(tuple, 3, false)/4",
    m = re.match("\((\S+), d(\d), (\S+)\)", args)
    hist = m.group(1)
    dim = int(m.group(2))
    cov = m.group(3)
    bench[(name, hist, dim, extra)].append((int(nbins) ** dim, x["cpu_time"]))

plt.figure(figsize=(7, 6))
handles = []
for (name, axis, cov), v in bench.items():
    v = np.sort(v).T
    # if "semi_dynamic" in axis: continue
    if cov == "all": continue
    lw = 3 if "Indexed" in name else 1.5
    col = {"Naive": "r", "Insider": "C0", "Indexed": "k"}.get(name, "k")
    ls = {"tuple": "-", "vector": "--", "vector_of_variant": ":"}[axis]
    name2 = {"Naive": "nested for (naive)", "Insider" : "nested for (opt.)", "Indexed": "indexed"}.get(name, name)
    h = plt.plot(v[0], v[1], lw=lw, ls=ls, color=col,
                 label=r"%s: ${\mathit{axes}}$ = %s" % (name2, axis))[0]
    handles.append(h)
handles.sort(key=lambda x: x.get_label())
plt.loglog()
plt.legend(handles=handles, fontsize="xx-small")
plt.ylabel("CPU time (less is better)")
plt.xlabel("number of bins in 3D histogram")
plt.tight_layout()
plt.savefig("iteration_performance.svg")
plt.show()
