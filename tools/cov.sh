#!/bin/sh
# must be executed in project root folder
lcov --rc lcov_branch_coverage=1 --base-directory `pwd`/test --directory `pwd`/../../bin.v2/libs/histogram/test --capture --output-file test.info
lcov --rc lcov_branch_coverage=1 --base-directory `pwd`/examples --directory `pwd`/../../bin.v2/libs/histogram/examples --capture --output-file examples.info

# merge files
lcov --rc lcov_branch_coverage=1 -a test.info -a examples.info -o all.info

# remove uninteresting entries
lcov --rc lcov_branch_coverage=1 --extract all.info "*/boost/histogram/*" --output-file coverage.info

# upload
curl -s https://codecov.io/bash | bash -s - -f coverage.info -X gcov
