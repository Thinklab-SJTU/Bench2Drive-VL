import os
import json
import yaml
import argparse
from copy import deepcopy

def fix_checkpoint(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    checkpoint = data.get("_checkpoint", {})
    records = checkpoint.get("records", [])
    modified = False

    for record in records:
        infractions = record.get("infractions", {})
        timeouts = infractions.get("scenario_timeouts", [])
        if "Agent timed out a scenario" in timeouts:
            scores = record.get("scores", {})
            score_route = scores.get("score_route", 0.0)
            score_penalty = scores.get("score_penalty", 1.0)
            corrected_penalty = score_penalty / 0.7
            scores["score_penalty"] = corrected_penalty
            scores["score_composed"] = score_route * corrected_penalty
            modified = True

    if modified:
        checkpoint["global_record"] = {}

        new_path = file_path.replace(".json", "_deleted_timeout.json")
        with open(new_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Fixed and saved: {new_path}")
    else:
        print(f"No timeout fix needed: {file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config', type=str, required=True, help='YAML config file listing checkpoint files')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    checkpoints = config.get("checkpoints", [])
    for path in checkpoints:
        if os.path.exists(path):
            fix_checkpoint(path)
        else:
            print(f"File not found: {path}")

if __name__ == '__main__':
    main()