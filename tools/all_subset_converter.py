import xml.etree.ElementTree as ET

xml_file = "./leaderboard/data/bench2drive220.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

routes_data = []

for route in root.findall("route"):
    route_id = route.get("id")
    
    for scenario in route.find("scenarios").findall("scenario"):
        scenario_type = scenario.get("type")
        scenario_name = scenario.get("name")
        
        routes_data.append((scenario_type, route_id, scenario_name))
routes_data.sort()

output_file = "all_routes.txt"

with open(output_file, "w") as f:
    for scenario_type, route_id, scenario_name in routes_data:
        line = f"type={scenario_type}, id={route_id}, name={scenario_name}\n"
        f.write(line)

print(f"Final result saved to {output_file}")
