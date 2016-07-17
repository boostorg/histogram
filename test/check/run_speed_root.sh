#!/bin/sh
g++ -std=c++11 -O3 `root-config --cflags` speed_root.cpp `root-config --libs` -lstdc++ -o /tmp/speed_root && /tmp/speed_root
