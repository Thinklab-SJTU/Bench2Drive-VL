import os
import argparse
import json
import yaml

def generate_config_and_script(cfg):
    # Create output directories if not exist
    os.makedirs(cfg["config_output_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    bash_lines = ["#!/bin/bash\n"]

    for name in cfg["subfolder_names"]:
        # Paths for subset and checkpoint files
        subset_file = os.path.join(cfg["config_output_dir"], f"subset_{name}.txt")
        checkpoint_file = os.path.join(cfg["config_output_dir"], f"finished_scenarios_{name}.txt")

        # Create empty files if not present
        for file_path in [subset_file, checkpoint_file]:
            if not os.path.exists(file_path):
                open(file_path, 'w').close()

        # Construct JSON config
        config = {
            "EVAL_SUBSET": cfg["eval_subset"],
            "USE_CHECKPOINT": cfg["use_checkpoint"],
            "SUBSET_FILE": subset_file,
            "CHECKPOINT_FILE": checkpoint_file,
            "INFERENCE_RESULT_DIR": os.path.join(cfg["inference_result_root"], name),
            "B2D_DIR": os.path.join(cfg["b2d_root"], name),
            "ORIGINAL_VQA_DIR": os.path.join(cfg["original_vqa_root"], name),
            "FRAME_PER_SEC": cfg["frame_per_sec"],
            "LOOK_FUTURE": cfg["look_future"]
        }

        # Write config JSON
        config_path = os.path.join(cfg["config_output_dir"], f"{name}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        # Build eval command
        out_dir = os.path.join(cfg["output_dir"], name)
        bash_line = (
            f"python ./B2DVL_Adapter/eval.py --config_dir {config_path} "
            f"--num_workers {cfg['num_workers']} "
            f"--out_dir {out_dir}"
        )

        # Add --num_sample if specified
        if "num_sample" in cfg:
            bash_line += f" --num_sample {cfg['num_sample']}"

        bash_lines.append(bash_line)

    # Write all bash commands to script
    bash_script_path = "run_all_evals.sh"
    with open(bash_script_path, "w") as f:
        f.write("\n".join(bash_lines))

    print(f"Generated {len(cfg['subfolder_names'])} configs and run_all_evals.sh.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config_file", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    generate_config_and_script(config)
