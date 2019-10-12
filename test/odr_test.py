import os
import sys
import re

this_path = os.path.dirname(__file__)

all_headers = set()
include_path = os.path.join(this_path, "..", "include")
for root, dirs, files in os.walk(include_path):
    for fn in files:
        fn = os.path.join(root,  fn)
        assert fn.startswith(include_path)
        fn = fn[len(include_path) :]
        all_headers.add(fn)


def get_headers(filename):
    with open(filename) as f:
        for hdr in re.findall('^#include [<"]([^>"]+)[>"]', f.read(), re.MULTILINE):
            if not hdr.startswith("boost/histogram"):
                continue
            yield hdr


included_headers = set()
unread_headers = set()
for hdr in get_headers(os.path.join(this_path, "/odr_test.cpp")):
    unread_headers.add(hdr)

while unread_headers:
    included_headers.update(unread_headers)
    for hdr in tuple(unread_headers):
        unread_headers.remove(hdr)
        for hdr2 in get_headers(include_path + hdr):
            if hdr2 not in included_headers:
                unread_headers.add(hdr2)

diff = sorted(all_headers - set(included_headers))

if not diff:
    sys.exit(0)


print("Header not included in odr_test.cpp:")
for fn in diff:
    print(fn)

sys.exit(1)
