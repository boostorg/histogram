import sys
import re
import glob
import os

def is_more_recent(a, b):
    return os.stat(a).st_mtime > os.stat(b).st_mtime

out_dir = os.path.dirname(__file__) + "/../examples"

exi = 1
for qbk in glob.glob(os.path.dirname(__file__) + "/*.qbk"):
    base = os.path.splitext(os.path.basename(qbk))[0]
    if base != "getting_started": continue
    with open(qbk) as fi:
        qbk_content = fi.read()
    qbk_needs_update = False
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
        foname = out_dir + "/%s_listing_%i.%s" % (base, exi, ext)
        if os.path.exists(foname):
            with open(foname) as fi:
                code2 = fi.read()
            if code != code2:
                if is_more_recent(qbk, foname):
                    with open(foname, "w") as fo:
                        fo.write(code)
                else:
                    qbk_content = qbk_content[:start] + code2 + qbk_content[end:]
                    qbk_needs_update = True
        else:
            with open(foname, "w") as fo:
                fo.write(code)
        exi += 1
    if qbk_needs_update:
        with open(qbk, "w") as fo:
            fo.write(qbk_content)
