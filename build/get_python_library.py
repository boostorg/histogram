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

library = d["LDLIBRARY"] or ("libpython" + sysconfig.get_python_version() + "*")
for libpath in ('LIBDEST', 'LIBDIR', 'LIBPL'):
    p = pj(d[libpath], library)
    if "*" in p:
        p = (glob(p) or [""])[0]
    if os.path.exists(p):
        sys.stdout.write(p)
        raise SystemExit

pprint("no library found, dumping config:")
pprint(d)
raise SystemExit(1)
