import os
import xml.etree.ElementTree as ET

subset_path = "subset.txt"
xml_path = "./leaderboard/data/bench2drive220.xml"
output_xml_path = "bench2drive_subset_routes.xml"
checkpoint_path = "subset_checkpoint.json"
startup_dir = "./leaderboard/scripts"
startup_script_path = os.path.join(startup_dir, "run_subset.sh")

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

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print(f"Deleted checkpoint file: {checkpoint_path}")
else:
    print("Checkpoint file not found, skipping deletion.")

os.makedirs(startup_dir, exist_ok=True)
with open(startup_script_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("BASE_PORT=22082\n")
    f.write("BASE_TM_PORT=50000\n")
    f.write("IS_BENCH2DRIVE=True\n")
    f.write("BASE_ROUTES=" + os.path.splitext(output_xml_path)[0] + "\n")
    f.write("TEAM_AGENT=leaderboard/team_code/data_agent.py\n")
    f.write("TEAM_CONFIG=your_team_agent_ckpt.pth\n")
    f.write("BASE_CHECKPOINT_ENDPOINT=" + os.path.splitext(checkpoint_path)[0] + "\n")
    f.write("SAVE_PATH=./eval_v1/\n")
    f.write("PLANNER_TYPE=only_traj\n")
    f.write("GPU_RANK=2\n")
    f.write("PORT=$BASE_PORT\n")
    f.write("TM_PORT=$BASE_TM_PORT\n")
    f.write("ROUTES=\"${BASE_ROUTES}.xml\"\n")
    f.write("CHECKPOINT_ENDPOINT=\"${BASE_CHECKPOINT_ENDPOINT}.json\"\n")
    f.write("export DEBUG=1\n")
    f.write("export SHOW_WP=1\n")
    f.write("export MINIMAL=1\n")
    f.write("bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK\n")

os.chmod(startup_script_path, 0o755)
print(f"Startup script generated: {startup_script_path}")