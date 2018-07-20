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
so_ext = so_ext[so_ext.rindex("."):]
library = "*python%s*%s" % (sysconfig.get_python_version(), so_ext) 
pprint(library)
for libpath in ('LIBDEST', 'LIBDIR', 'LIBPL'):
    p = pj(config[libpath], library)
    cand = glob(p)
    if cand and len(cand) == 1:
        sys.stdout.write(cand[0])
        raise SystemExit

pprint("no library found, dumping config:")
pprint(config)

for libpath in ('BINLIBDEST', 'LIBDEST', 'LIBDIR', 'LIBPL'):
    pprint(libpath)
    p = config[libpath]
    if os.path.exists(p):
        pprint(os.listdir(p))

raise SystemExit(1)
