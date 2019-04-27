import sys
import xml.etree.ElementTree as ET

tree = ET.parse(sys.argv[1])
root = tree.getroot()

parent_map = {c:p for p in tree.iter() for c in p}

# hide all unnamed template parameters, these are used for SFINAE
for item in root.iter("template-type-parameter"):
    if not item.get("name"):
        parent = parent_map[item]
        assert parent.tag == "template"
        parent.remove(item)

# replace any type with "detail" in its name with "implementation_defined"
for item in root.iter("type"):
    if not item.text:
        continue
    if "detail" in item.text:
        item.text = "implementation_defined"

# hide private member functions
for item in root.iter("method-group"):
    if item.get("name") == "private member functions":
        parent_map[item].remove(item)

# hide undocumented classes, structs, functions and replace those declared "implementation detail" with typedef to implementation_defined
for tag in ("class", "struct", "function"):
    for item in root.iter(tag):
        purpose = item.find("purpose")
        if purpose is None:
            parent_map[item].remove(item)
        elif purpose.text.strip().lower() == "implementation detail":
            name = item.get("name")
            item.clear()
            item.tag = "typedef"
            item.set("name", name)
            type = ET.Element("type")
            type.text = "implementation_defined"
            item.append(type)

# hide methods and constructors explicitly declared as "implementation detail"
for tag in ("constructor", "method"):
    for item in root.iter(tag):
        purpose = item.find("purpose")
        if purpose is not None and purpose.text.strip() == "implementation detail":
            parent_map[item].remove(item)

tree.write(sys.argv[2])
