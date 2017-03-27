import numpy as np
import matplotlib.pyplot as plt
import sys
import re
from collections import defaultdict, OrderedDict
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties

data = defaultdict(lambda:[])
for line in open("perf.dat"):
	if line and line[0] == "#": continue
	if line.isspace(): continue
	r = re.search("([0-9])D\n", line)
	if r:
		dim = int(r.group(1))
		continue
	if line.startswith("uniform"):
		dist = "uniform"
		continue
	if line.startswith("normal"):
		dist = "normal"
		continue
	label, time = line.strip().split(" ")
	time = float(time)
	data[dim].append((label, dist, time))

plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.12, right=0.92, top=0.95, bottom=0.1)
x = []
y = []
i = 0
for dim in sorted(data):
	v = data[dim]
	v2 = []
	labels = OrderedDict()
	for label, dist, time in v:
		if label in labels:
			labels[label].append(time)
		else:
			labels[label] = [time]
	j = 0
	for label,v in labels.items():
		tmin, tmax = sorted(v)
		i -= 1
		z = float(j) / len(labels)
		col = ((1.0-z) * np.array((1.0, 0.0, 0.0))
			   + z * np.array((1.0, 1.0, 0.0)))
		if label == "root":
			col = "k"
		if "numpy" in label:
			col = "0.6"
		if "gsl" in label:
			col = "0.3"
		# r1 = Rectangle((0, i), tmin, 1, facecolor=col, edgecolor="None")
		# r2 = Rectangle((0, i), tmax, 1, facecolor="None", edgecolor=col)
		# plt.gca().add_artist(r1)
		# plt.gca().add_artist(r2)
		r = Rectangle((0, i), 0.5 * (tmin + tmax), 1, facecolor=col)
		plt.gca().add_artist(r)
		tx = Text(-0.01, i+0.5, "%s" % label,
			      fontsize=17, va="center", ha="right", clip_on=False)
		plt.gca().add_artist(tx)
		j += 1
	i -= 1
	font0 = FontProperties()
	font0.set_size(20)
	font0.set_weight("bold")
	tx = Text(-0.01, i+0.6, "%iD" % dim,
			  fontproperties=font0, va="center", ha="right", clip_on=False)
	plt.gca().add_artist(tx)
plt.ylim(0, i)
plt.xlim(0, 1.0)

plt.tick_params("y", left="off", labelleft="off")
plt.xlabel("time (smaller is better)")

plt.savefig("benchmark.png")