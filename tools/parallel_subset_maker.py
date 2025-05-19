import os
import json
import yaml
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime

def generate_loop_script(gpu_id, output_name, startup_dir):
    now_str = datetime.now().strftime("%m%d_%H%M")
    logfile = f"{now_str}_{output_name}_{gpu_id}.log"
    script_name = f"loop_{output_name}_{gpu_id}.sh"
    script_path = os.path.join(startup_dir, script_name)
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"LOGFILE=\"{logfile}\"\n\n")
        f.write("while true; do\n")
        f.write("    echo \"Running at $(date)\" >> \"$LOGFILE\"\n")
        f.write(f"    bash ./{os.path.basename(startup_dir)}/{output_name}_run_{gpu_id}.sh >> \"$LOGFILE\" 2>&1\n")
        f.write("    echo \"Finished at $(date)\" >> \"$LOGFILE\"\n")
        f.write("    echo \"----------------------------------------\" >> \"$LOGFILE\"\n")
        f.write("done\n")
    os.chmod(script_path, 0o755)
    print(f"Loop script generated: {script_path}")

def generate_clean_script(gpu_id, output_name, startup_dir):
    base_port = 20082 + gpu_id * 1000
    port_start = base_port - 2
    port_end = base_port + 8
    script_name = f"clean_carla_{output_name}_{gpu_id}.sh"
    script_path = os.path.join(startup_dir, script_name)
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"for port in {{{port_start}..{port_end}}}; do\n")
        f.write("    echo \"Killing Carla on port $port...\"\n")
        f.write("    ps -ef | grep \"\\-carla-rpc-port=$port\" | grep -v grep | awk '{print $2}' | xargs -r kill -9\n\n")
        f.write("    echo \"Killing leaderboard on port $port...\"\n")
        f.write("    ps -ef | grep \"leaderboard\" | grep \"\\--port=$port\" | grep -v grep | awk '{print $2}' | xargs -r kill -9\n")
        f.write("done\n\n")
        f.write("wait\n")
    os.chmod(script_path, 0o755)
    print(f"Clean script generated: {script_path}")

def load_config(config_path):
    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            return json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        else:
            raise ValueError("Config file must be .json or .yaml")

def parse_filters(lines):
    filters = []
    for line in lines:
        line = line.strip()
        if line:
            conditions = dict(item.split("=") for item in line.split(", "))
            filters.append(conditions)
    return filters

def filter_routes(filters, xml_path):
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
    return filtered_routes

def write_xml(routes, output_path):
    new_root = ET.Element("routes")
    for route in routes:
        new_root.append(route)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_path, encoding="utf-8", xml_declaration=True)

def generate_script(gpu_id, output_name, startup_dir, config_path, env_vars=None):
    port = f"2{gpu_id:01}082"
    tm_port = f"5{gpu_id:01}000"
    base_routes = os.path.join(startup_dir, f"{output_name}_subset_routes_{gpu_id}")
    checkpoint_path = os.path.join(startup_dir, f"{output_name}_subset_checkpoint_{gpu_id}")

    script_path = os.path.join(startup_dir, f"{output_name}_run_{gpu_id}.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"BASE_PORT={port}\n")
        f.write(f"BASE_TM_PORT={tm_port}\n")
        f.write("IS_BENCH2DRIVE=True\n")
        f.write(f"BASE_ROUTES={base_routes}\n")
        f.write("TEAM_AGENT=leaderboard/team_code/data_agent.py\n")
        f.write("TEAM_CONFIG=your_team_agent_ckpt.pth\n")
        f.write(f"BASE_CHECKPOINT_ENDPOINT={checkpoint_path}\n")
        f.write("SAVE_PATH=./eval_v1/\n")
        f.write("PLANNER_TYPE=only_traj\n")
        f.write(f"GPU_RANK={gpu_id}\n")
        f.write(f"VLM_CONFIG={config_path}\n")
        f.write("PORT=$BASE_PORT\n")
        f.write("TM_PORT=$BASE_TM_PORT\n")
        f.write("ROUTES=\"${BASE_ROUTES}.xml\"\n")
        f.write("CHECKPOINT_ENDPOINT=\"${BASE_CHECKPOINT_ENDPOINT}.json\"\n")

        if env_vars:
            for key, val in env_vars.items():
                f.write(f"export {key}={val}\n")

        f.write("bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK $VLM_CONFIG\n")
    os.chmod(script_path, 0o755)
    print(f"Script generated: {script_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config", required=True, help="Path to config file (json/yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    subset_path = config["subset_path"]
    xml_path = config["xml_path"]
    output_name = config["output_name"]
    startup_dir = config.get("startup_dir", "./startup")
    gpu_ids = config["gpu_ids"]
    config_path = config["config_path"]

    os.makedirs(startup_dir, exist_ok=True)

    with open(subset_path, "r") as f:
        all_lines = f.readlines()

    gpu_count = len(gpu_ids)
    gpu_subsets = {gpu_id: [] for gpu_id in gpu_ids}
    for idx, line in enumerate(all_lines):
        target_gpu = gpu_ids[idx % gpu_count]
        gpu_subsets[target_gpu].append(line)

    for gpu_id in gpu_ids:
        subset_lines = gpu_subsets[gpu_id]
        subset_file_path = os.path.join(startup_dir, f"{output_name}_subset_{gpu_id}.txt")
        with open(subset_file_path, "w") as f:
            f.writelines(subset_lines)
        print(f"Wrote {subset_file_path} with {len(subset_lines)} lines")

        filters = parse_filters(subset_lines)
        filtered_routes = filter_routes(filters, xml_path)
        output_xml_path = os.path.join(startup_dir, f"{output_name}_subset_routes_{gpu_id}.xml")
        write_xml(filtered_routes, output_xml_path)
        print(f"Wrote {output_xml_path} with {len(filtered_routes)} routes")

        generate_script(gpu_id, output_name, startup_dir, config_path, env_vars=config.get("env_vars"))
        generate_loop_script(gpu_id, output_name, startup_dir)
        generate_clean_script(gpu_id, output_name, startup_dir)

if __name__ == "__main__":
    main()
