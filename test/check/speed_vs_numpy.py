#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from timeit import default_timer as timer
from histogram import histogram, regular_axis

def compare_1d(n, distrib):
    if distrib == 0:
        r = np.random.rand(n)
    else:
        r = 0.3 * np.random.randn(n)

    best_numpy = float("infinity")
    best_boost = float("infinity")
    for k in xrange(10):
        t = timer()
        w, xe = np.histogram(r, bins=100, range=(0.0, 1.0))
        t = timer() - t
        best_numpy = min(t, best_numpy)

        h = histogram(regular_axis(100, 0, 1))
        t = timer()
        h.fill(r)
        t = timer() - t
        best_boost = min(t, best_boost)

    print "1D"
    print "t[numpy] = %.3f" % best_numpy
    print "t[boost] = %.3f" % best_boost

def compare_3d(n, distrib):
    if distrib == 0:
        r = np.random.rand(3 * n)
    else:
        r = 0.3 * np.random.randn(3 * n)
    r = r.reshape(n, 3)

    best_numpy = float("infinity")
    best_boost = float("infinity")
    for k in xrange(10):
        t = timer()
        w, xe = np.histogramdd(r, bins=(100, 100, 100),
                               range=((0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0)))
        t = timer() - t
        best_numpy = min(t, best_numpy)

        h = histogram(regular_axis(100, 0, 1),
                      regular_axis(100, 0, 1),
                      regular_axis(100, 0, 1))
        t = timer()
        h.fill(r)
        t = timer() - t
        best_boost = min(t, best_boost)

    print "3D"
    print "t[numpy] = %.3f" % best_numpy
    print "t[boost] = %.3f" % best_boost

def compare_6d(n, distrib):
    if distrib == 0:
        r = np.random.rand(6 * n)
    else:
        r = 0.3 * np.random.randn(6 * n)
    r = r.reshape(n, 6)

    best_numpy = float("infinity")
    best_boost = float("infinity")
    for k in xrange(10):
        t = timer()
        w, xe = np.histogramdd(r, bins=(10, 10, 10,
                                        10, 10, 10),
                               range=((0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0)))
        t = timer() - t
        best_numpy = min(t, best_numpy)

        h = histogram(regular_axis(10, 0, 1),
                      regular_axis(10, 0, 1),
                      regular_axis(10, 0, 1),
                      regular_axis(10, 0, 1),
                      regular_axis(10, 0, 1),
                      regular_axis(10, 0, 1))
        t = timer()
        h.fill(r)
        t = timer() - t
        best_boost = min(t, best_boost)

    print "6D"
    print "t[numpy] = %.3f" % best_numpy
    print "t[boost] = %.3f" % best_boost

print "uniform distribution"
compare_1d(12000000, 0)
compare_3d(4000000, 0)
compare_6d(2000000, 0)
print "normal distribution"
compare_1d(12000000, 1)
compare_3d(4000000, 1)
compare_6d(2000000, 1)
