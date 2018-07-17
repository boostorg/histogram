import sys
from distutils import sysconfig

sys.stdout.write("using python : {version} : {prefix} : {inc} ;\n".format(
                 version=sysconfig.get_python_version(),
                 prefix=sysconfig.get_config_var("prefix"),
                 inc=sysconfig.get_python_inc()))
