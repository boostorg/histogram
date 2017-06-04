import sys
import re
import glob
import os

out_dir = os.getcwd() + "/doc_test"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

exi = 0
for qbk in glob.glob(os.path.dirname(__file__) + "/*.qbk"):
    base = os.path.splitext(os.path.basename(qbk))[0]
    for m in re.finditer("\[([^\]]+)\]``\s*([^`]+)``",
                         open(qbk).read()):
        tag = m.group(1)
        code = m.group(2)
        if tag == "c++":
            ext = "cpp"
        elif tag == "python":
            ext = "py"
        else:
            raise NotImplementedError("can only handle tags c++ and python")
        open(out_dir + "/%s_%i.%s" % (base, exi, ext), "w").write(code)
        exi += 1
