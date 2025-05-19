import json
import os
from io_utils import *
from collections import deque

DEFAULT_TASK_CONFIGS = {
    'INFER_SUBSET': True, # True if just inference a subset mentioned in configs/subset.txt
    'USE_CHECKPOINT': True, # True if record the finished scenarios and do not infer them next time
    'SUBSET_FILE': './infer_configs/subset.txt',
    'CHECKPOINT_FILE': './infer_configs/finished_scenarios.txt',
    'ENTRY_EXIT_FILE': './infer_configs/entry_exits.json',
    "FRAME_PER_SEC": 10 # the collection rate of data
}

DEFAULT_INFERENCE_BASICS = {
    'INPUT_WINDOW': 1, # 1 if only use current frame's input, n if uses n frames
    'CONVERSATION_WINDOW': 1, # 1 if only save conversation of current frame, n for n frames
    'USE_ALL_CAMERAS': False, # True if use all six, False if only use CAMERA_FRONT
    'NO_HISTORY_MODE': False, # True if do not save conversation history at all, which destorys the chain
    'APPEND_QUESTION': True, # True if use behaviour questions in configs/append_qustions.json
    'APPENDIX_FILE': './infer_configs/append_questions.json'
}

DEFAULT_CARLA_TASK_CONFIGS = {
    "FRAME_PER_SEC": 10 # the collection rate of data
}

DEFAULT_CARLA_INFERENCE_BASICS = {
    'INPUT_WINDOW': 1, # 1 if only use current frame's input, n if uses n frames
    'CONVERSATION_WINDOW': 1, # 1 if only save conversation of current frame, n for n frames
    'USE_ALL_CAMERAS': False, # True if use all six, False if only use CAMERA_FRONT
    'USE_BEV': True, # True if use bev image as input but not RGB cameras.
    'NO_HISTORY_MODE': False, # True if do not save conversation history at all, which destorys the chain
}

# qid can be found in hierachy_now.md
DEFAULT_CHAIN =  {
    "NODE": [19, 24, 27, 28, 10, 12, 8, 13, 15, 7, 39, 41, 42],
    "EDGE": {
        "19": [24, 27, 8, 10],
        "24": [28],
        "27": [28],
        "28": [8, 10, 12],
        "10": [13],
        "12": [13],
        "8": [42],
        "13": [42],
        "15": [7, 42],
        "7": [42],
        "39": [42],
        "41": [42],
        "42": []
    },
    "INHERIT": {
        "19": [8, 13, 7]
    },
    "USE_GT": [24, 27]
}

DEFAULT_PORT = 7023

class InferConfig:
    def __init__(self, config_path=None, in_carla=False):
        # Initialize default configurations
        self.TASK_CONFIGS = DEFAULT_TASK_CONFIGS if not in_carla else DEFAULT_CARLA_TASK_CONFIGS
        self.INFERENCE_BASICS = DEFAULT_INFERENCE_BASICS if not in_carla else DEFAULT_CARLA_INFERENCE_BASICS
        self.CHAIN = DEFAULT_CHAIN
        self.port = DEFAULT_PORT
        self.use_base64 = False
        self.order = []
        self.prev = {}
        self.in_carla = in_carla
        
        # Load configurations from the provided file path
        if config_path:
            self.load_config(config_path)
        self.CHAIN["EDGE"] = {int(k): list(map(int, v)) for k, v in self.CHAIN["EDGE"].items()}
        self.CHAIN["INHERIT"] = {int(k): list(map(int, v)) for k, v in self.CHAIN["INHERIT"].items()}
        self.preprocess_chain()
        self.CHAIN['ORDER'] = self.order
        self.CHAIN['PREV'] = self.prev
        # print_green(f"CHAIN_GRAPH_STRUCT = {self.CHAIN}")
    
    def topological_sort(self, nodes, edges):
        in_degree = {node: 0 for node in nodes}
        prev = {node: [] for node in nodes}
        
        for src, dests in edges.items():
            for dest in dests:
                if dest in in_degree:
                    in_degree[dest] += 1
                    prev[dest].append(src)
                else:
                    print_warning(f"Edge to undefined node '{dest}', ignoring.")
        
        queue = deque([node for node in nodes if in_degree[node] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            
            for next_node in edges.get(node, []):
                in_degree[next_node] -= 1
                if in_degree[next_node] == 0:
                    queue.append(next_node)

        if len(order) != len(nodes):
            return [], {}
        return order, prev

    def preprocess_chain(self):
        nodes = self.CHAIN.get("NODE", [])
        edges = self.CHAIN.get("EDGE", {})
        
        self.order, self.prev = self.topological_sort(nodes, edges)
        if not self.order:
            print_error("Cycle detected in CHAIN, using default configuration.")
            self.CHAIN = DEFAULT_CHAIN
            self.order, self.prev = self.topological_sort(
                DEFAULT_CHAIN.get("NODE", []), DEFAULT_CHAIN.get("EDGE", {}))
    
    def load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        with open(config_path, 'r') as file:
            try:
                loaded_config = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from file: {e}")

        if 'TASK_CONFIGS' in loaded_config:
            for key in self.TASK_CONFIGS.keys():
                if key in loaded_config['TASK_CONFIGS']:
                    self.TASK_CONFIGS[key] = loaded_config['TASK_CONFIGS'][key]
                else:
                    print_warning(f"Warning: Key '{key}' not found in TASK_CONFIGS, " +\
                                  f"using default value: '{self.TASK_CONFIGS[key]}'.")
        else:
            print_warning(f"Warning: TASK_CONFIGS not found in config file, " +\
                            f"using default values'.")
        
        if 'INFERENCE_BASICS' in loaded_config:
            for key in self.INFERENCE_BASICS.keys():
                if key in loaded_config['INFERENCE_BASICS']:
                    self.INFERENCE_BASICS[key] = loaded_config['INFERENCE_BASICS'][key]
                else:
                    print_warning(f"Warning: Key '{key}' not found in INFERENCE_BASICS, " +\
                                  f"using default value: '{self.INFERENCE_BASICS[key]}'.")
        else:
            print_warning(f"Warning: INFERENCE_BASICS not found in config file, " +\
                            f"using default values'.")
        
        if 'CHAIN' in loaded_config:
            if isinstance(loaded_config['CHAIN'], dict):
                if "NODE" not in self.CHAIN or "EDGE" not in self.CHAIN:
                    print_warning(f"Warning: CHAIN structure is not right, " +\
                            f"using default values'.")
                elif not isinstance(loaded_config['CHAIN']['NODE'], list) or \
                    not isinstance(loaded_config['CHAIN']['EDGE'], dict):
                    print_warning(f"Warning: CHAIN structure is not right, " +\
                            f"using default values'.")
                else:
                    self.CHAIN = loaded_config['CHAIN']
            else:
                print_warning("Warning: CHAIN must be a dict. Using default value.")
        else:
            print_warning(f"Warning: CHAIN not found in config file, " +\
                            f"using default values'.")
        
        if 'PORT' in loaded_config and self.in_carla:
            self.port = loaded_config['PORT']
        
        if 'USE_BASE64' in loaded_config:
            self.use_base64 = loaded_config['USE_BASE64']
        
        # no need.
        # if self.INFERENCE_BASICS['INPUT_WINDOW'] < self.INFERENCE_BASICS['CONVERSATION_WINDOW']:
            
        #     print_error(f"Warning: INPUT_WINDOW({self.INFERENCE_BASICS['INPUT_WINDOW']}) must be no less than " +\
        #                 f"CONVERSATION_WINDOW({self.INFERENCE_BASICS['CONVERSATION_WINDOW']}), changing all of them to " +\
        #                 f"{self.INFERENCE_BASICS['CONVERSATION_WINDOW']}.")
        #     self.INFERENCE_BASICS['INPUT_WINDOW'] = self.INFERENCE_BASICS['CONVERSATION_WINDOW']

    def display_config(self):
        print(f"TASK_CONFIGS: {self.TASK_CONFIGS}")
        print(f"INFERENCE_BASICS: {self.INFERENCE_BASICS}")
        print(f"CHAIN: {self.CHAIN}")
        if self.in_carla:
            print(f"PORT: {self.port}")

    def save_configs(self):
        data = {}
        data['TASK_CONFIGS'] = self.TASK_CONFIGS
        data['INFERENCE_BASICS'] = self.INFERENCE_BASICS
        data['CHAIN'] = self.CHAIN
        write_json(data, './infer_configs/saved_config.json')