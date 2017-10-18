import sys
import re
import glob
import os

def is_older(a, b):
    return os.stat(a).st_mtime < os.stat(b).st_mtime

out_dir = os.path.dirname(__file__) + "/../examples"

exi = 1
for qbk in glob.glob(os.path.dirname(__file__) + "/*.qbk"):
    base = os.path.splitext(os.path.basename(qbk))[0]
    with open(qbk) as fi:
        for m in re.finditer("\[([^\]]+)\]``\s*([^`]+)``", fi.read()):
            tag = m.group(1)
            code = m.group(2)
            if tag == "c++":
                ext = "cpp"
            elif tag == "python":
                ext = "py"
            else:
                raise NotImplementedError("can only handle tags c++ and python")
            foname = out_dir + "/%s_listing_%i.%s" % (base, exi, ext)
            if os.path.exists(foname) and is_older(foname, qbk):
                with open(foname, "w") as fo:
                    fo.write(code)
            exi += 1
