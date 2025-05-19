import os
import xml.etree.ElementTree as ET

subset_path = "subset.txt"
xml_path = "./leaderboard/data/bench2drive220.xml"
output_xml_path = "bench2drive_tmp_subset_routes.xml"

filters = []
with open(subset_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            conditions = dict(item.split("=") for item in line.split(", "))
            filters.append(conditions)

tree = ET.parse(xml_path)
root = tree.getroot()

filtered_routes = []
for route in root.findall("route"):
    route_id = route.get("id")
    scenario_types = {s.get("type") for s in route.findall(".//scenario")}
    scenario_names = {s.get("name") for s in route.findall(".//scenario")}
    
    for condition in filters:
        if ("id" not in condition or condition["id"] == route_id) and \
           ("type" not in condition or condition["type"] in scenario_types) and \
           ("name" not in condition or condition["name"] in scenario_names):
            filtered_routes.append(route)
            break

new_root = ET.Element("routes")
for route in filtered_routes:
    new_root.append(route)

new_tree = ET.ElementTree(new_root)
new_tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
print(f"Filtered routes saved to {output_xml_path}")
