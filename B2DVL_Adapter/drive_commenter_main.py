import argparse
from carla_vqa_generator import QAsGenerator
import string
import random
import pathlib
import json
import os
import glob
import concurrent.futures
import shutil
from collections import defaultdict
from generator_modules import *

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def parse_arguments():
    parser = argparse.ArgumentParser(description="QA Generator for Bench2Drive-VL, Augmented from DriveLM Carla")

    # Dataset and path settings
    path_group = parser.add_argument_group('Dataset and Path Settings')
    # path_group.add_argument('--base-folder', type=str, default='database',
    #                         help='Base folder for dataset')
    path_group.add_argument('--path-keyframes', type=str, required=False, default='path/to/keyframes.txt',
                            help='Path to the keyframes.txt')
    path_group.add_argument('--data-directory', type=str, required=True, default='path/to/dataset/data',
                            help='Data directory containing the dataset')
    path_group.add_argument('--output-graph-directory', type=str, required=True, default='output/path/of/graph/vqas',
                            help='Output directory for the vqa-graph')
    path_group.add_argument('--output-graph-examples-directory', type=str, default='output/path/of/graph/vqas',
                            help='Output directory for examples of the vqa-graph')
    path_group.add_argument('--path-maps', type=str, required=True, default='path/to/maps',
                            help='Data directory containings xodrs of CARLA maps')

    # Image and camera parameters
    img_group = parser.add_argument_group('Image and Camera Parameters')
    img_group.add_argument('--target-image-size', nargs=2, type=int, default=[1600, 900],
                           help='Target image size [width, height]')
    img_group.add_argument('--original-image-size', nargs=2, type=int, default=[1600, 900],
                           help='Original image size [width, height]')
    img_group.add_argument('--original-fov', type=float, default=110,
                           help='Original field of view')

    # Region of interest (ROI) for image projection
    roi_group = parser.add_argument_group('Region of Interest (ROI) Parameters')
    roi_group.add_argument('--min-y', type=int, default=0,
                           help='Minimum Y coordinate for ROI (to cut part of the bottom)')
    roi_group.add_argument('--max-y', type=int, default=None,
                           help='Maximum Y coordinate for ROI (to cut part of the bottom)')

    # Sampling parameters
    sampling_group = parser.add_argument_group('Sampling Parameters')
    sampling_group.add_argument('--random-subset-count', type=int, default=-1,
                                help='Number of random samples to use (-1 for all samples)')
    sampling_group.add_argument('--sample-frame-mode', choices=['all', 'keyframes', 'uniform'], default='uniform',
                                help='Frame sampling mode')
    sampling_group.add_argument('--sample-uniform-interval', type=int, default=5,
                                help='Interval for uniform sampling (if sample-frame-mode is "uniform")')

    # Visualization and saving options
    viz_group = parser.add_argument_group('Visualization and Saving Options')
    viz_group.add_argument('--save-examples', action='store_true', default=False,
                           help='Save example images')
    viz_group.add_argument('--visualize-projection', action='store_true', default=False,
                           help='Visualize object centers & bounding boxes in the image')
    viz_group.add_argument('--filter-routes-by-result', action='store_true', default=False,
                           help='Skip routes based on expert driving results')
    viz_group.add_argument('--remove-pedestrian-scenarios', action='store_true', default=False,
                           help='Skip scenarios with pedestrians')
    
    # Other Settings
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument('--worker-count', type=int, required=True, default=8,
                            help='Parallel Workers for Generation')

    args = parser.parse_args()

    # Compute derived parameters
    args.min_x = args.original_image_size[0] // 2 - args.target_image_size[0] // 2
    args.max_x = args.original_image_size[0] // 2 + args.target_image_size[0] // 2
    if args.max_y is None:
        args.max_y = args.target_image_size[1]

    return args

def load_marked_paths(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return set(file.read().splitlines())
    return set()

def worker_task(worker_index, scenario_subset, args):
    qas_generator = QAsGenerator(args, worker_index=worker_index, scenario_subset=scenario_subset)
    qas_generator.in_carla = False
    return qas_generator.create_qa_pairs(args.output_graph_directory)

if __name__ == '__main__':
    args = parse_arguments()

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    do_subset = int(os.environ.get('SUBSET', 0))
    if do_subset:
        subset_keys = load_marked_paths(os.environ.get('SUBSET_PATH', './subset.txt'))

    data_scenario_paths = [
        path for path in glob.glob(os.path.join(args.data_directory, "*"))
        if ("_" in os.path.basename(path) and os.path.isdir(path)) or path.endswith(".tar.gz")
    ]

    # to memorize progress
    processed_paths_file = os.environ.get('PROCESSED_PATH', "processed_paths.txt")
    processed_paths = load_marked_paths(processed_paths_file)
    data_scenario_paths = [x for x in data_scenario_paths if not (any(processed_path in x for processed_path in processed_paths))]

    if do_subset:
        data_scenario_paths = [x for x in data_scenario_paths if (any(marked_path in x for marked_path in subset_keys))]

    worker_count = args.worker_count
    scenario_splits = defaultdict(list)
    for index, path in enumerate(data_scenario_paths):
        scenario_splits[index % worker_count].append(path)

    stats_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_worker = {
            executor.submit(worker_task, i, scenario_splits[i], args): i for i in range(worker_count)
        }
    
        for future in concurrent.futures.as_completed(future_to_worker):
            llava_entry, stat_dict = future.result()
            stats_list.append(stat_dict)

    stats_dict = {
        'num_frames': 0,
        'min_num_questions': 999999999,
        'avg_num_questions': 0,
        'num_questions': 0,
        'num_objects': 0, 
        'num_questions_per_category': {},
        'stats_p3': {},
    }


    for sdict in stats_list:
        if (not isinstance(sdict, dict)) or sdict['num_questions'] <= 0:
            continue
        for key in sdict.keys():
            if key in ['num_frames', 'num_questions', 'num_objects']:
                stats_dict[key] += sdict[key]
        stats_dict['min_num_questions'] = min(stats_dict['min_num_questions'], sdict['min_num_questions'])
        for title_key in ['num_questions_per_category', 'stats_p3']:
            for key in sdict[title_key].keys():
                if key not in stats_dict[title_key]:
                    stats_dict[title_key][key] = sdict[title_key][key]
                else:
                    stats_dict[title_key][key] += sdict[title_key][key]

    stats_dict['avg_num_questions'] = stats_dict['num_questions'] / stats_dict['num_frames']

    with open(os.path.join(args.output_graph_directory, f'stats.json'), 'w', encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=4)
    print("Stats saved.")

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    
    print("All vqas generated. Congrats! (  = w = )")
