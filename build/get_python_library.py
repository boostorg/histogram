from distutils import sysconfig
import os
import sys
pj = os.path.join

d = sysconfig.get_config_vars()

for required_key in ("LDLIBRARY", "LIBDEST", "LIBDIR", "LIBPL"):
    if required_key not in d:
        raise StandardError("some keys not found:\n" + str(d))

library = d["LDLIBRARY"]
for libpath in ('LIBDEST', 'LIBDIR', 'LIBPL'):
    p = pj(d[libpath], library)
    if os.path.exists(p):
        sys.stdout.write(p)
        break
else:
    raise StandardError("no library found:\n" + str(d))
