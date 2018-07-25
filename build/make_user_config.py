import sys
from distutils import sysconfig

s = "using python : {version} : {prefix} : {inc} ;\n".format(
                 version=sysconfig.get_python_version(),
                 prefix=sysconfig.get_config_var("prefix"),
                 inc=sysconfig.get_python_inc())

sys.stdout.write(s)
