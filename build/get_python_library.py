from distutils import sysconfig
import os
import sys
import pprint
pj = os.path.join

d = sysconfig.get_config_vars()

for required_key in ("LDLIBRARY", "LIBDEST", "LIBDIR", "LIBPL"):
    if required_key not in d:
        pprint("some keys not found, dumping config:")
        pprint(d)
        raise SystemExit(1)

library = d["LDLIBRARY"]
for libpath in ('LIBDEST', 'LIBDIR', 'LIBPL'):
    p = pj(d[libpath], library)
    if os.path.exists(p):
        sys.stdout.write(p)
        raise SystemExit

pprint("no library found, dumping config:")
pprint(d)
raise SystemExit(1)
