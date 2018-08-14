# -*- coding: utf-8 -*-
#
# Copyright 2015-2016 Hans Dembinski
#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt
# or copy at http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
from timeit import default_timer as timer
from histogram import histogram
from histogram.axis import regular

def compare_1d(n, distrib):
    if distrib == 0:
        r = np.random.rand(n)
    else:
        r = 0.5 + 0.3 * np.random.randn(n)

    best_numpy = float("infinity")
    best_boost = float("infinity")
    for k in xrange(20):
        t = timer()
        w, xe = np.histogram(r, bins=100, range=(0.0, 1.0))
        t = timer() - t
        best_numpy = min(t, best_numpy)

        h = histogram(regular(100, 0, 1))
        t = timer()
        h.fill(r)
        t = timer() - t
        best_boost = min(t, best_boost)
    assert(np.all(w == np.array(h)[:-2]))

    print "py:numpy %.3f" % best_numpy
    print "py:hd_sd %.3f" % best_boost

def compare_2d(n, distrib):
    if distrib == 0:
        r = np.random.rand(n)
    else:
        r = 0.5 + 0.3 * np.random.randn(n)
    r = r.reshape(2, n/2)

    best_numpy = float("infinity")
    best_boost = float("infinity")
    for k in xrange(20):
        t = timer()
        w, xe, ye = np.histogram2d(r[0], r[1], bins=(100, 100),
                                   range=((0.0, 1.0), (0.0, 1.0)))
        t = timer() - t
        best_numpy = min(t, best_numpy)

        h = histogram(regular(100, 0, 1), regular(100, 0, 1))
        t = timer()
        h.fill(r[0], r[1])
        t = timer() - t
        best_boost = min(t, best_boost)
    assert(np.all(w == np.array(h)[:-2,:-2]))

    print "py:numpy %.3f" % best_numpy
    print "py:hd_sd %.3f" % best_boost

def compare_3d(n, distrib):
    if distrib == 0:
        r = np.random.rand(n)
    else:
        r = 0.3 * np.random.randn(n)
    r = r.reshape(3, n/3)

    best_numpy = float("infinity")
    best_boost = float("infinity")
    for k in xrange(20):
        t = timer()
        w, xe = np.histogramdd(r.T, bins=(100, 100, 100),
                               range=((0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0)))
        t = timer() - t
        best_numpy = min(t, best_numpy)

        h = histogram(regular(100, 0, 1),
                      regular(100, 0, 1),
                      regular(100, 0, 1))
        t = timer()
        h.fill(r[0], r[1], r[2])
        t = timer() - t
        best_boost = min(t, best_boost)
    assert(np.all(w == np.array(h)[:-2,:-2,:-2]))

    print "py:numpy %.3f" % best_numpy
    print "py:hd_sd %.3f" % best_boost

def compare_6d(n, distrib):
    if distrib == 0:
        r = np.random.rand(n)
    else:
        r = 0.3 * np.random.randn(n)
    r = r.reshape(6, n/6)

    best_numpy = float("infinity")
    best_boost = float("infinity")
    for k in xrange(20):
        t = timer()
        w, xe = np.histogramdd(r.T, bins=(10, 10, 10,
                                        10, 10, 10),
                               range=((0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0),
                                      (0.0, 1.0)))
        t = timer() - t
        best_numpy = min(t, best_numpy)

        h = histogram(regular(10, 0, 1),
                      regular(10, 0, 1),
                      regular(10, 0, 1),
                      regular(10, 0, 1),
                      regular(10, 0, 1),
                      regular(10, 0, 1))
        t = timer()
        h.fill(r[0], r[1], r[2], r[3], r[4], r[5])
        t = timer() - t
        best_boost = min(t, best_boost)
    assert(np.all(w == np.array(h)[:-2,:-2,:-2,:-2,:-2,:-2]))

    print "py:numpy %.3f" % best_numpy
    print "py:hd_sd %.3f" % best_boost

nfill = 6000000

print "1D"
print "uniform distribution"
compare_1d(nfill, 0)
print "normal distribution"
compare_1d(nfill, 1)

print "2D"
print "uniform distribution"
compare_2d(nfill, 0)
print "normal distribution"
compare_2d(nfill, 1)

print "3D"
print "uniform distribution"
compare_3d(nfill, 0)
print "normal distribution"
compare_3d(nfill, 1)

print "6D"
print "uniform distribution"
compare_6d(nfill, 0)
print "normal distribution"
compare_6d(nfill, 1)
