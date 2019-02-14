# must be executed in test folder
mkdir -p cov
cp -r ../../../bin.v2/libs/histogram/test/* cov
bash <(curl -s https://codecov.io/bash) -f '!*.cpp'
