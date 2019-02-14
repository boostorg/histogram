#!/bin/sh
# must be executed in project root folder
if [ -z $GCOV ]; then
  GCOV=gcov
fi
LCOV_OPTS="--gcov-tool=${GCOV} --rc lcov_branch_coverage=1"

lcov $LCOV_OPTS --base-directory `pwd`/test --directory `pwd`/../../bin.v2/libs/histogram/test --capture --output-file test.info
lcov $LCOV_OPTS --base-directory `pwd`/examples --directory `pwd`/../../bin.v2/libs/histogram/examples --capture --output-file examples.info

# merge files
lcov $LCOV_OPTS -a test.info -a examples.info -o all.info

# remove uninteresting entries
lcov $LCOV_OPTS --extract all.info "*/boost/histogram/*" --output-file coverage.info

# upload
curl -s https://codecov.io/bash | bash -s - -f coverage.info -X gcov -x $GCOV
