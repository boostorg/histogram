import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import re

data = json.load(open("doc/iteration.perf"))

bench = defaultdict(lambda:[])
for x in data["benchmarks"]:
    if x["run_type"] != "aggregate":
        continue
    if x["aggregate_name"] != "mean":
        continue
    name, arg = x["run_name"].split("/")
    m = re.match("(\S+)<(\S+), *(\S+)>", name)
    name = m.group(1)
    hist = m.group(2)
    extra = m.group(3)
    bench[(name, hist, extra)].append((int(arg) ** 3, x["cpu_time"]))

plt.figure(figsize=(7, 6))
handles = []
for (name, axis, extra), v in bench.items():
    v = np.sort(v).T
    # if "semi_dynamic" in axis: continue
    if "LessNaive" in name: continue
    if extra == "false": continue
    lw = 3 if "Indexed" in name else 1.5
    col = {"NaiveForLoop": "r", "InsiderForLoop": "C0", "IndexedLoop": "k"}.get(name, "k")
    ls = {"static_tag": "-", "semi_dynamic_tag": "--", "full_dynamic_tag": ":"}[axis]
    name2 = {"NaiveForLoop": "nested for (naive)", "InsiderForLoop" : "nested for (opt.)", "IndexedLoop": "indexed"}.get(name, name)
    axis2 = {"static_tag": "tuple", "semi_dynamic_tag": "vector",
             "full_dynamic_tag": "vector of variant"}.get(axis, axis)
    h = plt.plot(v[0], v[1], lw=lw, ls=ls, color=col,
                 label=r"%s: ${\mathit{axes}}$ = %s" % (name2, axis2))[0]
    handles.append(h)
handles.sort(key=lambda x: x.get_label())
plt.loglog()
plt.legend(handles=handles, fontsize="xx-small")
plt.ylabel("CPU time (less is better)")
plt.xlabel("number of bins in 3D histogram")
plt.tight_layout()
plt.savefig("iteration_performance.svg")
plt.show()
