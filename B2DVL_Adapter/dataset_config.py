import json
import os
from io_utils import *
from collections import deque

DATA_CONFIGS = {
    "SURROUND": True,
    "SUBSET": False,
    "NO_TAGS": True,
    "SUBSET_FILE": "./infer_configs/dataset_subset.txt",
    "ENTRY_EXIT_FILE": "./infer_configs/entry_exits.json",
    "INPUT_DIR": "../Carla_Chain_QA/carla_vqa_gen/vqa_dataset/outgraph",
    "OUTPUT_DIR": "./dataset",
    # "B2D_DIR": "/data/Bench2Drive_Data/Bench2Drive-Base-Tar",
    "B2D_DIR": "../Bench2Drive-rep"
}

# GRAPH_CONFIGS = {
#     "NODE": [19, 24, 27, 28, 10, 12, 8, 13, 15, 7, 39, 41, 42],
#     "EDGE": {
#         "19": [24, 27, 8, 10],
#         "24": [28],
#         "27": [28],
#         "28": [8, 10, 12],
#         "10": [13],
#         "12": [13],
#         "8": [42],
#         "13": [42],
#         "15": [7, 42],
#         "7": [42],
#         "39": [42],
#         "41": [42],
#         "42": []
#     },
#     "INHERIT": {
#         "19": [8, 13, 7]
#     },
#     "VALID": [42]
# }

# GRAPH_CONFIGS  =  {
#     "NODE": [19, 24, 27, 47, 10, 12, 8, 13, 15, 7, 39, 41, 42, 43, 44],
#     "EDGE": {
#         "19": [24, 27, 8, 10],
#         "24": [47, 42],
#         "27": [47, 42],
#         "47": [8, 10, 12],
#         "10": [13],
#         "12": [13],
#         "8": [43],
#         "13": [43],
#         "15": [7, 8],
#         "7": [8, 43],
#         "39": [43],
#         "41": [43],
#         "44": [43],
#         "43": [42],
#         "42": []
#     },
#     "INHERIT": {
#         "19": [43, 7],
#         "15": [7]
#     },
#     "VALID": [19, 24, 27, 47, 10, 12, 8, 13, 15, 7, 39, 41, 42, 43, 44],
# }

# stage 1
# GRAPH_CONFIGS = {
#     "NODE": [19, 15],
#     "EDGE": {
#         "19": [],
#         "15": []
#     },
#     "INHERIT": {
#         "19": [43, 7],
#         "15": [7]
#     },
#     "VALID": [19, 15]
# }

# stage 2
# GRAPH_CONFIGS = {
#     "NODE": [19, 15, 24, 7],
#     "EDGE": {
#         "19": [24],
#         "15": [7],
#         "24": [],
#         "7": []
#     },
#     "INHERIT": {
#     },
#     "VALID": [24, 7]
# }

# stage 3
# GRAPH_CONFIGS = {
#     "NODE": [19, 13, 24, 47],
#     "EDGE": {
#         "19": [24, 13, 47],
#         "24": [13, 47],
#         "13": [47],
#         "47": []
#     },
#     "INHERIT": {
#     },
#     "VALID": [13, 47]
# }

# stage 4
# GRAPH_CONFIGS = {
#     "NODE": [19, 24, 47, 15, 7, 8, 13, 43],
#     "EDGE": {
#         "19": [24, 13, 47, 8, 43],
#         "24": [13, 47, 8, 43],
#         "13": [47, 8, 43],
#         "47": [8, 43],
#         "15": [8, 43],
#         "7": [8, 43],
#         "8": [43],
#         "43": []
#     },
#     "INHERIT": {},
#     "VALID": [8, 43]
# }

# stage 5
GRAPH_CONFIGS = {
    "NODE": [19, 24, 47, 15, 7, 8, 13, 43, 50],
    "EDGE": {
        "19": [24, 47, 50],
        "24": [13, 47, 50],
        "47": [8, 50],
        "15": [8, 50],
        "7": [8, 50],
        "8": [43, 50],
        "13": [47, 50],
        "43": [50],
        "50": []
    },
    "INHERIT": {
        "19": [43, 7],
        "15": [7],
        "50": [7]
    },
    "VALID": [50]
}

FORMATS = ["middleware", "llava", "sharegpt", "sharegpt-CoT", "middleware-CoT"]

class DatasetConfig:
    def __init__(self):
        self.do_surround = DATA_CONFIGS['SURROUND']
        self.do_subset = DATA_CONFIGS['SUBSET']
        self.no_tags = DATA_CONFIGS['NO_TAGS']
        self.subset_file = DATA_CONFIGS['SUBSET_FILE']
        self.entry_exit_file = DATA_CONFIGS['ENTRY_EXIT_FILE']
        self.input_path = DATA_CONFIGS['INPUT_DIR']
        self.out_path = DATA_CONFIGS['OUTPUT_DIR']
        self.b2d_path = DATA_CONFIGS['B2D_DIR']
        self.formats = FORMATS

        self.CHAIN = GRAPH_CONFIGS
        self.CHAIN["EDGE"] = {int(k): list(map(int, v)) for k, v in self.CHAIN["EDGE"].items()}
        self.CHAIN["INHERIT"] = {int(k): list(map(int, v)) for k, v in self.CHAIN["INHERIT"].items()}
        self.preprocess_chain()
        self.CHAIN['ORDER'] = self.order
        self.CHAIN['PREV'] = self.prev
    
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

