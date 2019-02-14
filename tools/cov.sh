#!/bin/sh
# must be executed in project folder
mkdir -p test/cov
mkdir -p examples/cov
cp -r ../../bin.v2/libs/histogram/test/* test/cov
cp -r ../../bin.v2/libs/histogram/examples/* examples/cov
curl -s https://codecov.io/bash | bash -s -
