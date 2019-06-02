#!/usr/bin/env python3
from matplotlib import pyplot as plt, lines
import shelve
import json
import subprocess as subp
import sys
from collections import defaultdict
from run_benchmarks import get_commits, run
import numpy as np
import threading
threads = []
shelve_lock = threading.Lock()

commits, comments = get_commits()

def get_benchmarks(results):
    benchmarks = defaultdict(lambda: [])
    for hash in commits:
        if hash in results and results[hash] is not None:
            benchs = results[hash]
            for b in benchs["benchmarks"]:
                name = b["name"]
                time = min(b["cpu_time"], b["real_time"])
                benchmarks[name].append((commits.index(hash), time))
    return benchmarks

with shelve.open(sys.argv[1]) as results:
    benchmarks = get_benchmarks(results)

fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
plt.subplots_adjust(hspace=0, top=0.98, bottom=0.05, right=0.96)

plt.sca(ax[0])
for name, xy in benchmarks.items():
    if "uniform" in name: continue
    if "_1d" in name:
        x, y = np.transpose(xy)
        plt.plot(x, y, ".-", label=name, picker=5)
plt.legend(fontsize="xx-small")

plt.sca(ax[1])
for name, xy in benchmarks.items():
    if "uniform" in name: continue
    if "_2d" in name:
        x, y = np.transpose(xy)
        plt.plot(x, y, ".-", label=name, picker=5)
plt.legend(fontsize="xx-small")

plt.sca(ax[2])
for name, xy in benchmarks.items():
    if "uniform" in name: continue
    if "_3d" in name:
        x, y = np.transpose(xy)
        plt.plot(x, y, ".-", label=name, picker=5)
plt.legend(fontsize="xx-small")

plt.sca(ax[3])
for name, xy in benchmarks.items():
    if "uniform" in name: continue
    if "_6d" in name:
        x, y = np.transpose(xy)
        plt.plot(x, y, ".-", label=name, picker=5)
plt.legend(fontsize="xx-small")

plt.figtext(0.01, 0.5, "time per loop / ns [smaller is better]", rotation=90, va="center")

def format_coord(x, y):
    ind = max(0, min(int(x + 0.5), len(commits) - 1))
    hash = commits[ind]
    comment = comments[hash]
    return f"{hash} {comment}"

for axi in ax.flatten():
    axi.format_coord = format_coord

def onpick(event):
    thisline = event.artist
    ind = event.ind[0]
    hash = commits[int(thisline.get_xdata()[ind])]

    def worker(fig, ax, hash, lock):
        with lock:
            with shelve.open(sys.argv[1]) as results:
                del results[hash]
                run(results, comments, hash)
                benchmarks = get_benchmarks(results)

            for name in benchmarks:
                bench = benchmarks[name]
                _, y = bench[ind]
                for axi in ax.flatten():
                    for artist in axi.get_children():
                        if isinstance(artist, lines.Line2D) and artist.get_label() == name:
                            ydata = artist.get_ydata()
                            ydata[ind] = y
                            artist.set_ydata(ydata)
            fig.canvas.draw()

    print("updating", hash)
    t = threading.Thread(target=worker, args=(fig, ax, hash, shelve_lock))
    threads.append(t)
    t.start()

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
