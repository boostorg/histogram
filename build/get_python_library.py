from distutils import sysconfig
import os
import sys
from pprint import pprint
from glob import glob
pj = os.path.join

d = sysconfig.get_config_vars()

for required_key in ("LDLIBRARY", "LIBDEST", "LIBDIR", "LIBPL"):
    if required_key not in d:
        pprint("some keys not found, dumping config:")
        pprint(d)
        raise SystemExit(1)

library = "libpython" + sysconfig.get_python_version() + "*"
for libpath in ('BINLIBDEST', 'LIBDEST', 'LIBDIR', 'LIBPL'):
    p = pj(d[libpath], library)
    cand = glob(p)
    if cand and len(cand) == 1:
        sys.stdout.write(cand[0])
        raise SystemExit

pprint("no library found, dumping config:")
pprint(d)
raise SystemExit(1)
