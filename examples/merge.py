#!/usr/bin/env python
# -*- coding: utf-8 -*-
from icecube.histogram import I3HistogramN
from icecube import dataio, icetray
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        nargs="+")
    parser.add_argument("-o",
                        metavar="output",
                        default="merged.i3.gz",
                        help="name of output file")
    args = parser.parse_args()

    histograms = {}
    for fname in args.input:
        print "Processing:", fname
        frame = dataio.I3File(fname).pop_frame()
        for key in frame.keys():
            print "  -", key
            h = frame[key]
            if key not in histograms:
                histograms[key] = I3HistogramN(h)
            else:
                histograms[key] += h

    print "Writing:", args.o
    frame = icetray.I3Frame()
    for key in histograms:
	print "  -", key
        frame[key] = histograms[key]
    f = dataio.I3File(args.o, "w")
    f.push(frame)
    f.close()

if __name__ == "__main__":
    main()
