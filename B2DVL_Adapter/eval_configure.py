import json
import os
from io_utils import *

DEFAULT_EVAL_CONFIGS = {
    'EVAL_SUBSET': True, # True if just inference a subset mentioned in configs/subset.txt
    'USE_CHECKPOINT': True, # True if record the finished scenarios and do not infer them next time
    'SUBSET_FILE': './eval_configs/subset.txt',
    'CHECKPOINT_FILE': './eval_configs/finished_scenarios.txt',
    'INFERENCE_RESULT_DIR': '../infer_results/infer_res_for_test',
    'B2D_DIR': '../Bench2Drive-rep',
    "ORIGINAL_VQA_DIR": "../Carla_Chain_QA/carla_vqa_gen/vqa_dataset/outgraph",
    'FRAME_PER_SEC': 10, # the collection rate of data
    'LOOK_FUTURE': False, # True if allowing vlm to get answer for some questions earlier than gt, eg. traffic signs
}

class EvalConfig:
    def __init__(self, config_path=None):
        # Initialize default configurations
        self.CONFIGS = DEFAULT_EVAL_CONFIGS
        
        # Load configurations from the provided file path
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        with open(config_path, 'r') as file:
            try:
                loaded_config = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from file: {e}")
        
        for key in self.CONFIGS.keys():
            if key in loaded_config:
                self.CONFIGS[key] = loaded_config[key]
            else:
                print_warning(f"Warning: Key '{key}' not found in CONFIGS, " +\
                                f"using default value: '{self.CONFIGS[key]}'.")

    def display_config(self):
        print(f"CONFIGS: {self.CONFIGS}")

    def save_configs(self):
        write_json(self.CONFIGS, './eval_configs/saved_config.json')