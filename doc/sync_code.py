import sys
import re
import glob
import os

def is_more_recent(a, b):
    return os.stat(a).st_mtime > os.stat(b).st_mtime

out_dir = os.path.dirname(__file__) + "/../examples"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for qbk in glob.glob(os.path.dirname(__file__) + "/*.qbk"):
    base = os.path.splitext(os.path.basename(qbk))[0]
    with open(qbk) as fi:
        qbk_content = fi.read()
    exi = 1
    for m in re.finditer("\[([^\]]+)\]``\n*", qbk_content):
        tag = m.group(1)
        start = m.end()
        end = qbk_content.find("``", start)
        code = qbk_content[start:end]
        if tag == "c++":
            ext = "cpp"
        elif tag == "python":
            ext = "py"
        else:
            raise NotImplementedError("can only handle tags c++ and python")
        foname = out_dir + "/%s_listing_%02i.%s" % (base, exi, ext)
        if os.path.exists(foname):
            with open(foname) as fi:
                code2 = fi.read()
            if code != code2:
                if is_more_recent(qbk, foname):
                    with open(foname, "w") as fo:
                        fo.write(code)
        else:
            with open(foname, "w") as fo:
                fo.write(code)
        exi += 1
