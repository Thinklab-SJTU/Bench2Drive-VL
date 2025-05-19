import os
import json
from tqdm import tqdm
from waypoint_encoder import *

DATASET_ROOT = "../outgraph"
FRAME_RATE = 10
ENTRY = int(0.5 * FRAME_RATE)
TAIL = int(4 * FRAME_RATE)
KEYS = [
    '0.5s', '1.0s', '1.5s', '2.0s', '2.5s', '3.0s', '3.5s', '4.0s'
]
OUT_PATH = './all_waypoints.json'

def extract_delta_and_token_from_json(wp_json):
    key_len = len(KEYS)
    deltas = []
    delta_x = round(wp_json[KEYS[0]][0])
    delta_y = round(wp_json[KEYS[0]][1])
    deltas.append([delta_x, delta_y])
    for i in range (0, key_len - 1):
        delta_x = round(wp_json[KEYS[i + 1]][0] - wp_json[KEYS[i]][0], 2)
        delta_y = round(wp_json[KEYS[i + 1]][1] - wp_json[KEYS[i]][1], 2)
        deltas.append([delta_x, delta_y])
    xy_tokens = generate_motion_tokens(deltas)
    ds_tokens = generate_motion_and_direction_tokens(deltas)
    return deltas, xy_tokens, ds_tokens

def process_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        wp_answer = find_wp_answer(data)
        if wp_answer is not None:
            wp_json = json.loads(wp_answer)
            deltas, xy_tokens, ds_tokens = extract_delta_and_token_from_json(wp_json)
            return {
                'original_string': wp_json,
                'deltas': deltas,
                'xy_tokens': xy_tokens,
                'ds_tokens': ds_tokens
            }
        return None
    except Exception as e:
        print(f"Error occured when processing waypoint data.\nError message: {e}")
        return None

def find_wp_answer(vqa_json):
    res = None
    if "QA" in vqa_json:
        qa_json = vqa_json["QA"]
        if "behaviour" in qa_json:
            for qdict in qa_json["behaviour"]:
                if qdict['qid'] == 42:
                    return qdict['A']
    return res

def get_waypoint_dict_seq_from_rel(rel_wp_list):
    timestamp = 0.0
    x, y = 0.0, 0.0
    ans = {}
    for rel_wp in rel_wp_list:
        timestamp = round(timestamp + 0.5, 1)
        x += rel_wp[0]
        y += rel_wp[1]
        ans[f'{timestamp}s'] = [x, y]
    return ans

def main():
    folders = [
        f for f in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, f)) and "_" in f
    ]
    folders.sort()
    all_waypoints = []
    
    for folder in tqdm(folders):
        print(f"Processing {folder}...")
        folder_path = os.path.join(DATASET_ROOT, folder)
        
        json_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".json")
        ]
        json_files.sort()

        indices = [int(f.split('.')[0]) for f in json_files]
        if not indices:
            continue
        min_index, max_index = min(indices), max(indices)
        
        for json_file in json_files:
            index = int(json_file.split('.')[0])
            if index >= ENTRY and index < max_index - TAIL:
                json_path = os.path.join(folder_path, json_file)
                wp_dict = process_json(json_path)
                if wp_dict is not None:
                    all_waypoints.append(wp_dict)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as out_file:
        json.dump(all_waypoints, out_file, indent=4, ensure_ascii=False)
    print(f"All waypoints saved to {OUT_PATH}")

if __name__ == "__main__":
    main()