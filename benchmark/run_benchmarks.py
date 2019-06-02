#!/usr/bin/env python3
"""
Run this from a special build directory that uses the benchmark folder as root

    cd my_build_dir
    cmake ../benchmark
    ../run_benchmarks.py

This creates a database, benchmark_results. Plot it:

    ../plot_benchmarks.py benchmark_results

The script leaves the include folder in a modified state. To clean up, do:

    git checkout HEAD -- ../include
    git clean -f -- ../include

"""
import subprocess as subp
import tempfile
import os
import shelve
import json
import sys


def get_commits():
    commits = []
    comments = {}
    for line in subp.check_output(("git", "log", "--oneline")).decode("ascii").split("\n"):
        if line:
            ispace = line.index(" ")
            hash = line[:ispace]
            commits.append(hash)
            comments[hash] = line[ispace+1:]
    commits = commits[::-1]
    return commits, comments


def recursion(results, commits, comments, ia, ib):
    ic = int((ia + ib) / 2)
    if ic == ia:
        return
    run(results, comments, commits[ic])
    if all([results[commits[i]] is None for i in (ia, ib, ic)]):
        return
    recursion(results, commits, comments, ia, ic)
    recursion(results, commits, comments, ic, ib)


def run(results, comments, hash):
    if hash in results:
        return
    print(hash, comments[hash])
    results[hash] = None
    subp.call(("rm", "-rf", "../include"))
    if subp.call(("git", "checkout", hash, "--", "../include")) != 0:
        sys.stderr.write("[Benchmark] Cannot checkout include folder\n")
        return
    print(hash, "make")
    with tempfile.TemporaryFile() as out:
        if subp.call(("make", "-j4", "histogram_filling"), stdout=out, stderr=out) != 0:
            sys.stderr.write("[Benchmark] Cannot make benchmarks\n")
            out.seek(0)
            sys.stderr.write(out.read().decode("utf-8") + "\n")
            return
    print(hash, "run")
    s = subp.check_output(("./histogram_filling", "--benchmark_format=json"))
    results[hash] = d = json.loads(s)
    for benchmark in d["benchmarks"]:
        print(benchmark["name"], min(benchmark["real_time"], benchmark["cpu_time"]))


def main():
    with shelve.open("benchmark_results") as results:
        commits, comments = get_commits()
        if len(sys.argv) == 2:
            # redo this commit
            hash = sys.argv[1]
            del results[hash]
            run(results, comments, hash)
        else:
            if len(sys.argv) == 3:
                first = sys.argv[1]
                last = sys.argv[2]
            else:
                first = commits[0]
                last = commits[-1]
            # retry first, last if previous builds failed
            if first in results and results[first] is None:
                del results[first]
            if last in results and results[last] is None:
                del results[last]
            run(results, comments, first)
            run(results, comments, last)
            recursion(results, commits, comments, commits.index(first), commits.index(last))

if __name__ == "__main__":
    main()
