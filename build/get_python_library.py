from distutils import sysconfig
import os
import sys
from pprint import pprint
from glob import glob
pj = os.path.join

config = sysconfig.get_config_vars()

for required_key in ("LDLIBRARY", "LIBDEST", "LIBDIR", "LIBPL"):
    if required_key not in config:
        pprint("some keys not found, dumping config:")
        pprint(config)
        raise SystemExit(1)

so_ext = config['SO']
library = "*python" + sysconfig.get_python_version() + "*" + so_ext 
for libpath in ('BINLIBDEST', 'LIBDEST', 'LIBDIR', 'LIBPL'):
    p = pj(d[libpath], library)
    cand = glob(p)
    if cand and len(cand) == 1:
        sys.stdout.write(cand[0])
        raise SystemExit

pprint("no library found, dumping config:")
pprint(d)

for libpath in ('BINLIBDEST', 'LIBDEST', 'LIBDIR', 'LIBPL'):
    pprint(libpath)
    if os.path.exists(libpath):
        pprint(os.listdir(libpath))

raise SystemExit(1)
