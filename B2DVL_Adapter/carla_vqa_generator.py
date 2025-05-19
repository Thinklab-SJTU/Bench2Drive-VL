import os
import glob
import gzip
import json
import tqdm
import random
import shutil
import tarfile
import numpy as np
from pathlib import Path
from collections import Counter
import cv2
from PIL import Image, ImageDraw
from collections import Counter
import carla
from generator_modules import *
import re
from io_utils import *

"""
Main class for processing and converting the Bench2Drive dataset to the Graph-QAs for Bench2Drive-VL.
"""

class QAsGenerator():
    all_qa_pairs = []

    def __init__(self, args, worker_index, scenario_subset):
        self.in_carla = os.environ.get('VQA_GEN', 0)
        self.worker_index = worker_index

        print(f"Worker {worker_index} gets scenarios: {scenario_subset}")

        # Image and camera parameters
        self.TARGET_IMAGE_SIZE = args.target_image_size
        self.ORIGINAL_IMAGE_SIZE = args.original_image_size
        self.ORIGINAL_FOV = args.original_fov

        # Region of interest (ROI) for image projection
        self.MIN_X = args.min_x
        self.MAX_X = args.max_x
        self.MIN_Y = args.min_y
        self.MAX_Y = args.max_y

        # Sampling parameters
        self.random_subset_count = args.random_subset_count
        self.sample_frame_mode = args.sample_frame_mode
        self.sample_uniform_interval = args.sample_uniform_interval

        # Visualization and saving options
        self.save_examples = args.save_examples         
        self.visualize_projection = args.visualize_projection 
        self.filter_routes_by_result = args.filter_routes_by_result
        self.remove_pedestrian_scenarios = args.remove_pedestrian_scenarios

        self.data_directory = args.data_directory 
        self.path_keyframes = args.path_keyframes 

        self.output_graph_directory = args.output_graph_directory 
        self.output_graph_examples_directory = args.output_graph_examples_directory

        # Build camera projection matrix
        self.CAMERA_MATRIX = None
        self.CAM_INTRINSIC_FRONT = None
        self.CAM_INTRINSIC_FRONT_LEFT = None
        self.CAM_INTRINSIC_FRONT_RIGHT = None
        self.CAM_INTRINSIC_BACK = None
        self.CAM_INTRINSIC_BACK_LEFT = None
        self.CAM_INTRINSIC_BACK_RIGHT = None
        self.WORLD2CAM_FRONT = None
        self.WORLD2CAM_FRONT_LEFT = None
        self.WORLD2CAM_FRONT_RIGHT = None
        self.WORLD2CAM_BACK = None
        self.WORLD2CAM_BACK_LEFT = None
        self.WORLD2CAM_BACK_RIGHT = None
        self.CAMERA_MATRIX = build_projection_matrix(self.ORIGINAL_IMAGE_SIZE[0],
                                                     self.ORIGINAL_IMAGE_SIZE[1],
                                                     self.ORIGINAL_FOV)
        self.CAM_DICT = {}

        # creaete the directories where we save the graph and some graph examples
        Path(self.output_graph_directory).mkdir(parents=True, exist_ok=True)
        if self.save_examples:
            Path(self.output_graph_examples_directory).mkdir(parents=True, exist_ok=True)

        # # all the paths to the boxes in the data
        # self.data_boxes_paths = glob.glob(os.path.join(self.data_directory, '**/anno/*.json.gz'), recursive=True)

        # # Randomly sample a subset of data (if random_subset_count > 0)
        # if self.random_subset_count > 0:
        #     random.shuffle(self.data_boxes_paths)
        #     self.data_boxes_paths = self.data_boxes_paths[:self.random_subset_count]

        # self.data_boxes_paths = list(sorted(self.data_boxes_paths))

        # get directories of all scenarios
        self.data_scenario_paths = scenario_subset

        if self.random_subset_count > 0:
            random.shuffle(self.data_scenario_paths)
            self.data_scenario_paths = self.data_scenario_paths[:self.random_subset_count]

        self.data_scenario_paths = list(sorted(self.data_scenario_paths))

        self.list_next_junction_id_minus_one = []

        # added for b2d
        self.map_file_dir = args.path_maps
        self.town_name = None
        self.map = None

        # camera infos
        self.CAMERA_FRONT = {}
        self.CAMERA_FRONT_LEFT = {}
        self.CAMERA_FRONT_RIGHT = {}
        self.CAMERA_BACK = {}
        self.CAMERA_BACK_LEFT = {}
        self.CAMERA_BACK_RIGHT = {}

        # to look into future
        self.current_measurement_path = None
        self.current_measurement_index = 0
        self.last_special_move_index = INVALID_NUM
        self.last_left_lane = INVALID_NUM
        self.scenario_ignore_interval = 30

        # for [debug]
        self.appended_measurements = {}

        # config
        self.frame_rate = 10 # collect 10 data every 1s

        # global variables for calculation
        self.inf_num = 999
        self.leftmost_pos_of_left_hazard = {}
        self.last_full_scenario_name = 'last_name'

        self.ego_command_str = 'follow the road'
        self.all_vehicles_info = {}

        # for traffic sign
        self.traffic_sign_map = {
            'stop_sign':  
            { 
                'type_id': 'traffic.stop',
                'visual_description': 'Stop sign',
                'detailed_description': 'the stop sign',
                'behaviour': 'should stop soon',
            },
            'speed_limit':
            { 
                'type_id': 'traffic.speed_limit',
                'visual_description': 'Speed limit sign',
                'detailed_description': 'the speed limit sign',
                'behaviour': 'should adjust its speed soon',
            },
            'accident_warning':
            { 
                'type_id': 'static.prop.warningaccident',
                'visual_description': 'Accident warning sign',
                'detailed_description': 'the accident warning sign',
                'behaviour': 'should slow down and maybe change lane later because of the accident ahead',
            },
            'construction_warning':
            { 
                'type_id': 'static.prop.warningconstruction',
                'visual_description': 'Construction warning sign',
                'detailed_description': 'the construction warning sign',
                'behaviour': 'should slow down and maybe change lane later because of the construction ahead',
            },
            'construction_cone':
            { 
                'type_id': 'static.prop.constructioncone',
                'visual_description': 'Construction cone',
                'detailed_description': 'the construction cone',
                'behaviour': 'should avoid knocking construction cone off',
            },
            'traffic_warning':
            {
                'type_id': 'static.prop.trafficwarning',
                'visual_description': 'Traffic warning sign',
                'detailed_description': 'the traffic warning sign',
                'behaviour': 'should slow down and maybe change lane later because of the obstacle ahead',
            },
            'yield_sign': # exists in 'InterurbanAdvancedActorFlow_Town13_Route686_Weather10'
            {
                'type_id': 'traffic.yield',
                'visual_description': 'Yield sign',
                'detailed_description': 'the yield sign',
                'behaviour': 'should drive cautiously and yield fast vehicles at the intersection',
            }
        }

        # for speed_limit calculation
        self.valid_speed_limits = []
        # if road changes, speed limit dies
        self.last_road_id = INVALID_NUM # random impossible number
        self.last_lane_id = INVALID_NUM
        self.opposite_flag = False # in TwoWays scenarios, vehicle is at opposite direction
        self.current_speed_limit = INF_MAX
        self.future_speed_limit = INF_MAX
        self.ahead_speed_limit = None
        self.passed_speed_limit = None
        self.first_lane_command = 4 # command initialized to follow_lane
        self.lane_clear_threshold = LANE_CLEAR_THRESHOLD_BASE # safe for lane changing
        self.lane_forward_threshold = LANE_FORWARD_THRESHOLD_BASE # also alert the vehicle ahead in targeted lane
        self.side_blind_spot_flag = False
        self.first_walker_position = None

        self.leave_highway_scenarios = ['HighwayExit', 'MergerIntoSlowTraffic', 'InterurbanActorFlow']
        self.enter_highway_scenarios = ['HighwayCutIn', 'MergerIntoSlowTrafficV2', 'InterurbanAdvancedActorFlow']
        self.highway_change_lane_scenarios = ['HighwayExit', 'MergerIntoSlowTraffic', 'MergerIntoSlowTrafficV2', 'InterurbanActorFlow', 'InterurbanAdvancedActorFlow']
        self.circumvent_scenarios = [
            'Accident',
            'AccidentTwoWays',
            'ConstructionObstacle',
            'ConstructionObstacleTwoWays',
            'HazardAtSideLane',
            'HazardAtSideLaneTwoWays',
            'ParkedObstacle',
            'ParkedObstacleTwoWays',
            'VehicleOpensDoor',
            'VehicleOpensDoorTwoWays',
        ]
        self.merging_scenarios = [
            'CrossingBicycleFlow', 'EnterActorFlow', 'NonSignalizedJunctionLeftTurn', 
            'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow',
            'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow',
            'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 
            'VehicleTurningRoute', 'VehicleTurningRoutePedestrian',
            'TJunction', 'VanillaNonSignalizedTurn', 
            'VanillaSignalizedTurnEncounterGreenLight',
            'VanillaSignalizedTurnEncounterRedLight', 
            'VanillaNonSignalizedTurnEncounterStopsign'
        ]
        self.right_merging_scenarios = [
            'NonSignalizedJunctionRightTurn', 'SignalizedJunctionRightTurn'
        ]
        self.vehicle_ids_following_ego = []
        self.role_vehicle_ids = []
        self.not_passed_circumvent_obstacles = True
        self.distance_to_circumvent_obstacle = None
        self.ideal_flow_speed = NORMAL_KEEP_SPEED
        
        self.answer43_changelane = ""
        self.answer43_brake = ""
        self.crossed_junction_flag = False

        self.merging_and_needs_accelerate = None
        self.merging_and_needs_stop = None
        self.merging_danger_vehicles = []
        self.stopped_at_stop_sign = False
        self.obstacle_blocked_lane_id = None

        # cache
        self.in_cache_list = []

        # process checkpoint
        self.processed_paths_file = os.environ.get('PROCESSED_PATH', "processed_paths.txt")

        self.accelerate_white_list = {}
        # used when the ego vehicle needs to take chance to merge/cross actor flow
        # the vehicle in the list will be ignored during a time interval (MERGING_IGNORE_INVERVAL)
        # because the ego vehicle must take a dangerous action
        # and vice versa...
        self.accelerate_black_list = {}
        # records the must stop reason vehicles generates by basic vector checking
        # value is their distances
        # if distance decreases and relative speed < 0 (approaching)
        # this vehicle is still considered as a 'must stop' reason.
        self.first_appear_intervals = {}
        # records interval threshold of actor flows
        self.front_merge_vehicle_ids = {}
        # records all vehicles cut in the ego vehicle's lane
        # in HighwayCutIn scenario, the cut in vehicle matches the ego vehicle's speed
        # which means if we stop, the scenario will timeout
        # so this scenario should be handled specially
    
    def load_processed_paths(self):
        return self.load_marked_paths(self.processed_paths_file)

    def load_marked_paths(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return set(file.read().splitlines())
        return set()

    def save_processed_path(self, path):
        with open(self.processed_paths_file, 'a') as file:
            file.write(path + '\n')

    def reset_qa_stats(self):
        # Initialize data structures
        self.vqa_llava_format = {'image_paths': [], 'llava_format': [], 'image_subset': []}
        self.min_num_questions = INF_MAX
        self.total_num_questions = 0
        self.total_num_objects = 0
        self.num_questions_per_category = {
            'parked_vehicles': 0,
            'dynamic_vehicles': 0,
            'roadlayout': 0,
            'trafficsign': 0,
            'trafficlight': 0,
            'environment': 0,
            'pedestrian': 0,
            'behaviour': 0,
            'ego': 0,
        }
        self.stats_p3 = {'perception': 0, 'planning': 0, 'prediction': 0, 'behaviour': 0}

        self.frame_num = 0
        self.skipped_frames = 0
    
    def process_single_frame(self, path, data, scenario_name, route_number, frame_number, output_graph_directory):
        self.current_measurement_index = int(frame_number)
        self.strict_mode = os.environ.get('STRICT_MODE', 1)

        if self.in_carla:
            self.save_name = data['save_name']
        else: 
            self.save_name = scenario_name
        # with gzip.open(path_measurements, 'rb') as f:
        #     file_content = f.read()
        #     measurements = json.loads(file_content.decode('utf-8'))

        # Get perception questions
        # print("generation reached here") # debug
            
        sensor_data = data['sensors']

        # Read all camera datas
        if 'CAM_FRONT' in sensor_data:
            self.CAMERA_FRONT = sensor_data['CAM_FRONT']
            self.CAM_INTRINSIC_FRONT = sensor_data['CAM_FRONT']['intrinsic']
            self.WORLD2CAM_FRONT = sensor_data['CAM_FRONT']['world2cam']
            self.CAMERA_MATRIX = self.CAM_INTRINSIC_FRONT
            self.CAM_DICT['CAM_FRONT'] = sensor_data['CAM_FRONT']
        if 'CAM_FRONT_LEFT' in sensor_data:
            self.CAMERA_FRONT_LEFT = sensor_data['CAM_FRONT_LEFT']
            self.CAM_INTRINSIC_FRONT = sensor_data['CAM_FRONT_LEFT']['intrinsic']
            self.WORLD2CAM_FRONT_LEFT = sensor_data['CAM_FRONT_LEFT']['world2cam']
            self.CAM_DICT['CAM_FRONT_LEFT'] = sensor_data['CAM_FRONT_LEFT']
        if 'CAM_FRONT_RIGHT' in sensor_data:
            self.CAMERA_FRONT_RIGHT = sensor_data['CAM_FRONT_RIGHT']
            self.CAM_INTRINSIC_FRONT = sensor_data['CAM_FRONT_RIGHT']['intrinsic']
            self.WORLD2CAM_FRONT_RIGHT = sensor_data['CAM_FRONT_RIGHT']['world2cam']
            self.CAM_DICT['CAM_FRONT_RIGHT'] = sensor_data['CAM_FRONT_RIGHT']
        if 'CAM_BACK' in sensor_data:
            self.CAMERA_BACK = sensor_data['CAM_BACK']
            self.CAM_INTRINSIC_FRONT = sensor_data['CAM_BACK']['intrinsic']
            self.WORLD2CAM_BACK = sensor_data['CAM_BACK']['world2cam']
            self.CAM_DICT['CAM_BACK'] = sensor_data['CAM_BACK']
        if 'CAM_BACK_LEFT' in sensor_data:
            self.CAMERA_BACK_LEFT = sensor_data['CAM_BACK_LEFT']
            self.CAM_INTRINSIC_FRONT = sensor_data['CAM_BACK_LEFT']['intrinsic']
            self.WORLD2CAM_BACK_LEFT = sensor_data['CAM_BACK_LEFT']['world2cam']
            self.CAM_DICT['CAM_BACK_LEFT'] = sensor_data['CAM_BACK_LEFT']
        if 'CAM_BACK_RIGHT' in sensor_data:
            self.CAMERA_BACK_RIGHT = sensor_data['CAM_BACK_RIGHT']
            self.CAM_INTRINSIC_FRONT = sensor_data['CAM_BACK_RIGHT']['intrinsic']
            self.WORLD2CAM_BACK_RIGHT = sensor_data['CAM_BACK_RIGHT']['world2cam']
            self.CAM_DICT['CAM_BACK_RIGHT'] = sensor_data['CAM_BACK_RIGHT']

        # print(f"[debug] path = {path}")
        image_path = path.replace('anno', 'camera/rgb_front').replace('.json.gz', '.jpg')
        relative_image_path = image_path
        # print(f"[debug] image_path = {image_path}")

        if self.in_carla:
            scenario_name = data['scenario_type']
            # print(f"[debug] scenario_name = {scenario_name}")
        
        data['junction_exit_wp_x'], data['junction_exit_wp_y'] = find_last_non_junction_waypoint(self.map,
                                                                                                 x=data['x_command_far'],
                                                                                                 y=data['y_command_far'])
        
        res = self.generate_perception_questions(data, scenario_name, scenario_name, route_number, frame_number)
        qas, num_questions, num_objects, questions_per_category, key_object_infos, extra_flags = res
        for key, values in qas.items():
            for value in values:
                self.stats_p3[value['type']] += 0.5  # We have questions and answers

        # Save examples if specified
        if self.visualize_projection:
            # Load and draw on the image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # original_img_path = f'{self.output_graph_examples_directory}/original_images/{scenario_name}'
            # Path(original_img_path).mkdir(parents=True, exist_ok=True)
            # resized_img_path = f'{self.output_graph_examples_directory}/resized_images/{scenario_name}'
            # Path(resized_img_path).mkdir(parents=True, exist_ok=True)
            anno_img_path = f'{self.output_graph_examples_directory}/anno_images/{self.save_name}'
            Path(anno_img_path).mkdir(parents=True, exist_ok=True)
            # graph_path = f'{self.output_graph_examples_directory}/graphs/{scenario_name}'
            # Path(graph_path).mkdir(parents=True, exist_ok=True)

            assert image.width == self.ORIGINAL_IMAGE_SIZE[0], f'{image.width} != {self.ORIGINAL_IMAGE_SIZE[0]}'
            assert image.height == self.ORIGINAL_IMAGE_SIZE[1], f'{image.height} != {self.ORIGINAL_IMAGE_SIZE[1]}'
            
            # Draw a point for each object (e.g, car, traffic light, ...) on the image
            ego_z = 0.0
            world2ego = None
            if self.WORLD2CAM_FRONT is not None:
                for single_object in data['bounding_boxes']:
                    if 'distance' in single_object and single_object['distance'] > 40.0: # too far, neglect
                        continue
                    # print(single_object)
                    if 'location' in single_object:
                        single_object['position'] = transform_to_ego_coordinates(single_object['center'], self.WORLD2CAM_FRONT)
                        if single_object['class'] == 'ego_vehicle':
                            ego_z = single_object['location'][2]
                            world2ego = single_object['world2ego']
                        all_points_2d, _ = project_all_corners(single_object, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)

                        if 'vehicle' in single_object['class']:
                            # for debug
                            # all_points_2d = []
                            # projected_corners, visibility = get_vehicle_projected_corners(self.CAMERA_FRONT, single_object)
                            # for idx, is_visible in enumerate(visibility):
                            #     if is_visible:
                            #         all_points_2d.append(projected_corners[idx])
                            # debug section ends
                            color = (255, 0, 0, 0)
                        elif 'traffic_light' in single_object['class'] or 'stop' in single_object['class'] or 'stop' in single_object['type_id']:
                            color = (0, 255, 0, 0)
                        elif 'landmark' in single_object['class']:
                            color = (0, 0, 255, 0)
                        else:
                            color = (0, 0, 0, 0)
                        if all_points_2d is not None and len(all_points_2d) > 0:
                            top_left_point = min(all_points_2d, key=lambda p: (p[0], p[1]))
                            
                            for points_2d in all_points_2d:
                                draw.ellipse((points_2d[0] - 5, points_2d[1] - 5, points_2d[0] + 5, points_2d[1] + 5), 
                                            fill=color)
                            
                            label = ""
                            if 'type_id' in single_object:
                                label = f"{label}{str(single_object['type_id'])}"
                            if 'id' in single_object:
                                label = f"{label}, id={str(single_object['id'])}"

                            draw.text((top_left_point[0] + 5, top_left_point[1] - 10), label, 
                                        fill=color)
                
                # if self.in_carla:
                #     relative_command_far = [data['x_command_far'], data['y_command_far'], 0.0]
                #     relative_command_near = [data['x_command_near'], data['y_command_near'], 0.0]
                #     absolute_command_far = transform_to_world_coordinates(relative_command_far, world2ego)
                #     absolute_command_near = transform_to_world_coordinates(relative_command_near, world2ego)
                #     print(f"[debug] self inversion = {transform_to_world_coordinates([0, 0, 0], world2ego)}")
                #     print(f"[debug] self x = {transform_to_world_coordinates([1, 0, 0], world2ego)}")
                #     print(f"[debug] self y = {transform_to_world_coordinates([0, 1, 0], world2ego)}")
                #     print(f"[debug] absolute_command_far = {absolute_command_far}")
                #     print(f"[debug] absolute_command_near = {absolute_command_near}")
                # else:
                absolute_command_far = [data['x_command_far'], data['y_command_far'], ego_z]
                absolute_command_near = [data['x_command_near'], data['y_command_near'], ego_z]

                WP_HEIGHT = 1.0

                absolute_command_far_top = (absolute_command_far[0], absolute_command_far[1], absolute_command_far[2] + WP_HEIGHT)
                absolute_command_near_top = (absolute_command_near[0], absolute_command_near[1], absolute_command_near[2] + WP_HEIGHT)

                far_command_uv_top = project_point(absolute_command_far_top, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                near_command_uv_top = project_point(absolute_command_near_top, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                
                # relative_command_far = transform_to_ego_coordinates(absolute_command_far, self.WORLD2CAM_FRONT)
                # relative_command_near = transform_to_ego_coordinates(absolute_command_near, self.WORLD2CAM_FRONT)
                # # print(f"[debug] relative_command_far = {relative_command_far}")
                # # print(f"[debug] relative_command_near = {relative_command_near}")
                far_command_uv = project_point(absolute_command_far, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                near_command_uv = project_point(absolute_command_near, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                color = (0, 0, 255, 0)

                if far_command_uv is not None:
                    draw.ellipse((far_command_uv[0] - 5, far_command_uv[1] - 5, far_command_uv[0] + 5, far_command_uv[1] + 5), fill=color)

                if near_command_uv is not None:
                    draw.ellipse((near_command_uv[0] - 5, near_command_uv[1] - 5, near_command_uv[0] + 5, near_command_uv[1] + 5), fill=color)

                if far_command_uv_top is not None:
                    label = f"{data['command_far']}: far_command_point"
                    draw.text((far_command_uv_top[0] + 5, far_command_uv_top[1] - 10), label, fill=color)

                if near_command_uv_top is not None:
                    label = f"{data['command_far']}: near_command_point"
                    draw.text((near_command_uv_top[0] + 5, near_command_uv_top[1] - 10), label, fill=color)
                
                if far_command_uv is not None and far_command_uv_top is not None:
                    draw.line([(far_command_uv_top[0], far_command_uv_top[1]), (far_command_uv[0], far_command_uv[1])], 
                              fill=color, width=2)

                if near_command_uv is not None and near_command_uv_top is not None:
                    draw.line([(near_command_uv_top[0], near_command_uv_top[1]), (near_command_uv[0], near_command_uv[1])], 
                              fill=color, width=2)
                
                if data['junction_exit_wp_x'] is not None and data['junction_exit_wp_y'] is not None:
                    absolute_junc_wp = (data['junction_exit_wp_x'], data['junction_exit_wp_y'], ego_z)
                    absolute_junc_wp_top = (data['junction_exit_wp_x'], data['junction_exit_wp_y'], ego_z + WP_HEIGHT)
                    junc_wp_uv_top = project_point(absolute_junc_wp_top, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                    junc_wp_uv = project_point(absolute_junc_wp, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                    if junc_wp_uv is not None:
                        draw.ellipse((junc_wp_uv[0] - 5, junc_wp_uv[1] - 5, junc_wp_uv[0] + 5, junc_wp_uv[1] + 5), fill=color)
                    if junc_wp_uv_top is not None:
                        label = f"junction exit"
                        draw.text((junc_wp_uv_top[0] + 5, junc_wp_uv_top[1] - 10), label, fill=color)
                    if junc_wp_uv is not None and junc_wp_uv_top is not None:
                        draw.line([(junc_wp_uv_top[0], junc_wp_uv_top[1]), (junc_wp_uv[0], junc_wp_uv[1])], 
                                fill=color, width=2)
                
            annotated_image_path = f'{anno_img_path}/{int(frame_number):05d}.png'
            image.save(annotated_image_path)
            
            # Save QA data (canceled, moved from outside function to below)
            # file_name = f'{self.output_graph_examples_directory}/graphs/' \
            #             f'{scenario_name}/{int(frame_number):05d}.json'
            # with open(file_name, 'w', encoding='utf-8') as f:
            #     json.dump(qas, f, sort_keys=True, indent=4)

        # TODO: implement an offline method to do stats, 
        # since we do not always generate full dataset in a row.
                
        # Update minimum number of questions
        self.min_num_questions = min(self.min_num_questions, num_questions)

        # Update statistics
        self.total_num_questions += num_questions
        self.total_num_objects += num_objects
        for key, value in questions_per_category.items():
            self.num_questions_per_category[key] += value

        # Append QA data to the list
        self.all_qa_pairs.append(qas)

        # Populate VQA LLAVA format data
        vqa_llava_entry = {}
        vqa_llava_entry['image'] = relative_image_path
        vqa_llava_entry['conversations'] = qas
        vqa_llava_entry['conversations']['key_object_infos'] = key_object_infos
        self.vqa_llava_format['image_paths'].append(relative_image_path)
        self.vqa_llava_format['llava_format'].append(vqa_llava_entry)

        self.frame_num += 1

        path, tick = relative_image_path, vqa_llava_entry

        # Generate a random tick ID
        # tick_id = generate_random_string()
        tick_data = {}
        # scene_data[scene_id]['key_frames'][tick_id] = tick_data

        # Store key object information
        tick_data['key_object_infos'] = tick['conversations']['key_object_infos']

        # Create a dictionary to store QA data
        qa_data = {}
        tick_data['QA'] = qa_data
        qa_data['perception'] = []
        qa_data['prediction'] = []
        qa_data['planning'] = []
        qa_data['behaviour'] = []

        # Store extra flags
        tick_data['extra_flags'] = extra_flags

        # Store image path
        image_path = tick['image']
        tick_data['image_paths'] = {key: None for key in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                                                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']}
        tick_data['image_paths']['CAM_FRONT'] = image_path
        tick_data['image_paths']['CAM_FRONT_LEFT'] = image_path.replace('front', 'front_left')
        tick_data['image_paths']['CAM_FRONT_RIGHT'] = image_path.replace('front', 'front_right')
        tick_data['image_paths']['CAM_BACK'] = image_path.replace('front', 'back')
        tick_data['image_paths']['CAM_BACK_LEFT'] = image_path.replace('front', 'back_left')
        tick_data['image_paths']['CAM_BACK_RIGHT'] = image_path.replace('front', 'back_right')


        for key, items in tick['conversations'].items():
            # Available keys: important_objects, dynamic_vehicles, roadlayout, stopsign, trafficlight, environment, ego
            if key == 'key_object_infos':
                continue

            for i in range(len(items) // 2):
                q_dic = items[2*i]
                a_dic = items[2*i + 1]
                qa_item = {
                    'Q': q_dic['value'],
                    'A': a_dic['value'],
                    'C': None,
                    'qid': q_dic['qid'],
                    'con_up': q_dic['connection_up'],
                    'con_down': q_dic['connection_down'],
                    'cluster': q_dic['chain'],
                    'layer': q_dic['layer']
                }
                if 'object_tags' in q_dic:
                    qa_item['object_tags'] = q_dic['object_tags']

                qa_data[q_dic['type']].append(qa_item)

        _, _, _, image_number = path.split('/')[-4:]

        image_number = image_number.replace('.jpg', '')
        
        save_dir = os.path.join(output_graph_directory, self.save_name)
        print(f"[debug] output_graph_directory = {output_graph_directory}, save_dir = {save_dir}, image_number = {image_number}")
        
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        with open(save_dir + '/' + image_number + '.json', 'w', encoding='utf-8') as f:
            json.dump(tick_data, f, indent=4, default=str)

        return tick_data

    def create_qa_pairs(self, output_graph_directory):
        """
        Create all question answer pairs in llava format, convert them to NuScenes afterwards and finally save them
        """
        self.reset_qa_stats()

        keyframes_list = None
        # Load keyframes list if sampling keyframes
        if self.sample_frame_mode == 'keyframes':
            # print("mode = keyframes, parsing keyframes.") # debug
            keyframes_list_path = self.path_keyframes
            with open(keyframes_list_path, 'r', encoding="utf-8") as f:
                keyframes_list = f.readlines()
            keyframes_list = [x.strip() for x in keyframes_list]
            keyframes_list = [x.replace('rgb', 'boxes').replace('.jpg', '.json.gz') for x in keyframes_list]

        # # For debug, used when wanting a subset
        # do_subset = int(os.environ.get('SUBSET', 0))
        # if do_subset:
        #     subset_keys = self.load_marked_paths(os.environ.get('SUBSET_PATH', './subset.txt'))
        
        scenario_count = len(self.data_scenario_paths)
        # Process each frame
        for sidx in range(0, scenario_count):
            scenario_path = self.data_scenario_paths[sidx]

            # put in main
            # # skip if path or path's root path is in processed_paths.txt
            # if any(processed_path in scenario_path for processed_path in self.processed_paths):
            #     # print(f"[info] skipping {path}, which has already marked processed.")
            #     continue

            # if do_subset:
            #     if not (any(marked_path in scenario_path for marked_path in subset_keys)):
            #         # print(f"[info] skipping {path}, which has already marked processed.")
            #         continue

            scenario_base_name = os.path.basename(scenario_path)
            scenario_base_name = scenario_base_name.replace(".tar.gz", "")
            if os.path.isdir(scenario_path):
                boxes_paths = sorted(glob.glob(os.path.join(scenario_path, "anno", "*.json.gz")))

            elif scenario_path.endswith(".tar.gz"):
                extract_dir = os.path.join(CACHE_DIR, scenario_base_name)
                os.makedirs(extract_dir, exist_ok=True)
                print(f"\033[32m{scenario_base_name} is compressed. Extracting to {extract_dir}\033[0m")

                self.in_cache_list.append(extract_dir)
                if len(self.in_cache_list) > CACHE_SIZE:
                    shutil.rmtree(self.in_cache_list[0])
                    self.in_cache_list = self.in_cache_list[1:]
                
                with tarfile.open(scenario_path, "r:gz") as tar:
                    tar.extractall(path=extract_dir)
                print(f"\033[32mExtracted.\033[0m")
                
                boxes_paths = sorted(glob.glob(os.path.join(extract_dir, scenario_base_name, "anno", "*.json.gz")))

            else:
                continue

            desc_str = f"[Worker {self.worker_index}] {scenario_base_name} ({sidx} / {scenario_count})"
            for path in tqdm.tqdm(boxes_paths, lock_args=False, desc=desc_str):
                # # skip if path or path's root path is in processed_paths.txt
                # if any(processed_path in path for processed_path in self.processed_paths):
                #     # print(f"[info] skipping {path}, which has already marked processed.")
                #     continue

                # if do_subset:
                #     if not (any(marked_path in path for marked_path in subset_keys)):
                #         # print(f"[info] skipping {path}, which has already marked processed.")
                #         continue

                # Skip frames based on keyframes list
                if self.sample_frame_mode == 'keyframes':
                    if path not in keyframes_list:
                        self.skipped_frames += 1
                        continue
                
                frame_number = int(path.split('/')[-1].split('.')[0])

                # Skip frames if sampling uniformly and frame number does not match
                if self.sample_frame_mode == 'uniform' and frame_number % self.sample_uniform_interval != 0:
                    continue

                # print("analysing path") # debug
                route_dir = '/'.join(path.split('/')[:-2])
                scenario_name = route_dir.split('/')[-1]
                town_name = route_dir.split('/')[-1].split('_')[1]
                route_number = route_dir.split('/')[-1].split('_')[0] + '_' + route_dir.split('/')[-1].split('_')[1] + '_' + route_dir.split('/')[-1].split('_')[2]

                self.current_measurement_path = path
                dir_name, file_name = os.path.split(path)
                base_name, _ = os.path.splitext(file_name)
                base_name, _ = os.path.splitext(base_name)
                # self.current_measurement_index = int(base_name)
                
                # print(f"[debug] path: {path}, route_dir: {route_dir}, scenario_name: {scenario_name}, town_name: {town_name}, route_number: {route_number}")
                if self.map is None or (self.town_name is None or self.town_name is not town_name):
                    self.town_name = town_name
                    if os.path.exists(os.path.join(self.map_file_dir, f'OpenDrive/{self.town_name}.xodr')):
                        with open(os.path.join(self.map_file_dir, f'OpenDrive/{self.town_name}.xodr'), 'r') as fp:
                            self.map = carla.Map('{self.town_name}', fp.read())
                    else:
                        with open(os.path.join(self.map_file_dir, f'{self.town_name}/OpenDrive/', f'{self.town_name}.xodr'), 'r') as fp:
                            self.map = carla.Map('{self.town_name}', fp.read())

                # # Skip this scenario because it is not annotated correctly
                # if 'InterurbanAdvancedActorFlow' in route_dir:
                #     continue

                # # Skip this scenario because it is not annotated correctly
                # # and we cannot differentiate between entry and exit properly
                # if 'MergerIntoSlowTraffic' in route_dir:
                #     continue

                # # Skip this scenario because language labels are not adjusted
                # if 'VehicleTurningRoute' == route_dir:
                #     continue

                # # Skip scenarios with pedestrians if specified
                # if self.remove_pedestrian_scenarios:
                #     if 'DynamicObjectCrossing' in route_dir:
                #         continue
                #     if 'ParkingCrossingPedestrian' in route_dir:
                #         continue
                #     if 'PedestrianCrossing' in route_dir:
                #         continue
                #     if 'VehicleTurningRoutePedestrian' in route_dir:
                #         continue

                # Skip frames if RGB image does not exist
                if not os.path.isfile(path.replace('anno', 'camera/rgb_front').replace('.json.gz', '.jpg')):
                    self.skipped_frames += 1
                    continue

                # Check if files exist
                if not os.path.exists(path):
                    continue
                # there's only 'anno' folder in B2D dataset.
                # if not os.path.exists(path_measurements):
                #     continue

                # Read data and measurements files
                with gzip.open(path, 'rb') as f:
                    file_content = f.read()
                    data = json.loads(file_content.decode('utf-8'))
                tick_data = self.process_single_frame(path=path,
                                                      data=data,
                                                      scenario_name=scenario_name,
                                                      route_number=route_number,
                                                      frame_number=frame_number,
                                                      output_graph_directory=output_graph_directory)
            
            self.save_processed_path(scenario_path)

        # TODO: implement an offline method to do stats, 
        # since we do not always generate full dataset in a row.
            
        # Save statistics
        stats_dict = {
            'num_frames': self.frame_num,
            'min_num_questions': self.min_num_questions,
            'avg_num_questions': self.total_num_questions / self.frame_num if self.frame_num > 0 else 0,
            'num_questions': self.total_num_questions, 
            'num_objects': self.total_num_objects, 
            'num_questions_per_category': self.num_questions_per_category,
            'stats_p3': self.stats_p3,
        }
        # with open(os.path.join(self.output_graph_directory, f'stats_{self.worker_index}.json'), 'w', encoding="utf-8") as f:
        #     json.dump(stats_dict, f, indent=4)
        # print(f"Stats {self.worker_index} saved.")

        # Convert and save VQA LLAVA format data
        return self.vqa_llava_format, stats_dict

    def project_object_center(self, obj):
        """
        Projects the provided objects center onto the 2D image plane.
        """
        object_position = obj['position']
        object_position_3d = np.array([object_position[1], -object_position[2], object_position[0]])
        rotation_vector = np.zeros((3, 1), np.float32)
        translation_vector = np.array([[0.0, 2.0, 1.5]], np.float32)

        # Define the distortion coefficients
        distortion_coeffs = np.zeros((5, 1), np.float32)

        object_center_2d, _ = cv2.projectPoints(object_position_3d, rotation_vector, translation_vector, 
                                                self.CAMERA_MATRIX, distortion_coeffs)
                
        return object_center_2d[0][0]

    def should_consider_vehicle(self, vehicle):
        """
        True, if it's in front cam
        False, if vehicle is not bicycle and the number of points on it are below a threshold
        False, if the vehicle is behind the ego vehicle
        False, if it's a parking vehicle, that does not cut in
        """
        # print(f"[debug] id = {vehicle['id']}, speed = {vehicle['speed']}, type_id = {vehicle['type_id']}, num_points = {vehicle['num_points']}")
        # If the vehicle is parked and not cutting in, exclude it from consideration
        if vehicle.get('state', 'null') == "static":
            if 'state' not in vehicle:
                print(f"[Waning] This vehicle doesn't have 'state' attribute: {vehicle}, set to null.")
            return False
        # Max. distance is 25m and similar to the max. distance of junctions
        if  (
            vehicle['position'][0] < BACK_CONSIDER_THRESHOLD
            or (vehicle['class'] != 'bicycle' and vehicle['num_points'] < MIN_OBJECT_NUM_POINT)
            or (vehicle['num_points'] < MIN_BICYCLE_NUM_POINT)
        ):
            # if vehicle['num_points'] < 10:
            #     print(f"[debug] not seen in lidar!")
            # else:
            #     print(f"[debug] position not good!")
            return False

        # Check if the vehicle is visible in the image
        # But here only vehicle in front is considered!
        # A pity...
        # but this is only used to identify obstacle in forward route
        # so it's ok
        vehicle_is_visible = is_vehicle_in_camera(self.CAMERA_FRONT, vehicle) # or \
                             # is_vehicle_in_camera(self.CAMERA_FRONT_LEFT, vehicle)
        # if vehicle_is_visible == False:
        #     print(f"[debug] not in camera!")
        return vehicle_is_visible

    def is_object_in_front(self, vehicle):
        """
        True, if it's in front cam
        False, if vehicle is not bicycle and the number of points on it are below a threshold
        False, if the vehicle is behind the ego vehicle
        False, if it's a parking vehicle, that does not cut in
        """
        if vehicle['state'] == "static":
            # print('[debug] is static, so no consider')
            return False
        # Max. distance is 25m and similar to the max. distance of junctions
        if  (
            vehicle['position'][1] > FRONT_Y_OFFSET
            or vehicle['position'][1] < -FRONT_Y_OFFSET
            or vehicle['position'][0] < BACK_CONSIDER_THRESHOLD
            or (vehicle['class'] != 'bicycle' and vehicle['num_points'] < MIN_OBJECT_NUM_POINT)
            or (vehicle['num_points'] < MIN_BICYCLE_NUM_POINT)
        ):
            return False

        # Check if the vehicle is visible in the image
        # But here only vehicle in front is considered!
        # A pity...
        # but this is only used to identify obstacle in forward route
        # so it's ok
        vehicle_is_visible = is_vehicle_in_camera(self.CAMERA_FRONT, vehicle)

        return vehicle_is_visible
    
    def get_obstacle_pass_offset(self, scenario_name):
        offset = 0
        if 'Accident' in scenario_name:
            offset = ACCIDENT_PASS_OFFSET
        if 'Construction' in scenario_name:
            offset = CONSTRUCTION_PASS_OFFSET
        if 'HazardAtSideLane' in scenario_name:
            offset = HAZARDATSIDE_PASS_OFFSET
        return offset

    def generate_2d_box_from_projected_points(self, projected_points):
        return [
                [round(projected_points[:, 0].min(), 1), round(projected_points[:, 1].min(), 1)],
                [round(projected_points[:, 0].max(), 1), round(projected_points[:, 1].max(), 1)],
            ]

    def generate_object_key_value(self, id, category, visual_description, detailed_description, object_count, 
                                  is_role=False, is_dangerous=False, obj_dict=None,
                                  projected_dict=None):
        """
        Generate a key-value pair representing an object detected in an image, including its category,
        visual description, 2D bounding box coordinates, and position.
        """
        # Create a dictionary to store the object's information
        object_info = {
            'id': id,
            'Category': category,
            'Status': None,
            'Visual_description': visual_description,
            'Detailed_description': detailed_description,
            '2d_bbox': {},
            '3d_bbox': {},
            'is_role': is_role,
            'is_dangerous': is_dangerous,
            'obj_dict': obj_dict,
            'position': obj_dict.get('position') if obj_dict is not None else None,
            'distance': obj_dict.get('distance') if obj_dict is not None else None,
            'speed': obj_dict.get('position') if obj_dict is not None else None
        }

        # Add the 2D bounding box coordinates, if available
        tag_str = ""
        if projected_dict is not None and len(projected_dict) > 0:
            for tag_dict in projected_dict:
                projected_points, projected_points_meters = tag_dict['projected_points'], tag_dict['projected_points_meters']
                cam_key = tag_dict['cam_key']
                object_info['2d_bbox'][cam_key] = self.generate_2d_box_from_projected_points(projected_points)
                object_info['3d_bbox'][cam_key] = projected_points_meters.round(1).tolist()

                mean = np.round(np.mean(object_info['2d_bbox'][cam_key], axis=0), decimals=1)
                center_x = float(mean[0])
                center_y = float(mean[1])

                tag_str += f"<{cam_key},{center_x},{center_y}>"

        # Generate a unique key for the object
        if id is None:
            if tag_str != "":
                object_key = f"<e{object_count + 1}{tag_str}>"
            else:
                object_key = f"<e{object_count + 1}<CAM_FRONT>>"
        else:
            if tag_str != "":
                object_key = f"<c{id}{tag_str}>"
            else:
                object_key = f"<c{id}<CAM_FRONT>>"

        return object_key, object_info

    def add_qas_questions(self, qa_list, qid, chain, layer, qa_type, connection_up, connection_down, 
                          question, answer, object_id=None, object_tags=[]):
        question = question.replace("(['", "(")
        question = question.replace("'])", ")")
        question = question.replace("([])", "")
        question = question.replace("()", "")

        answer = answer.replace("(['", "(")
        answer = answer.replace("'])", ")")
        answer = answer.replace("([])", "")
        answer = answer.replace("()", "")

        answer = re.sub(r'\s+', ' ', answer).strip() # replace bad formats, eg. begin and end spaces, multi-spaces
        
        qa_list.append({'qid': qid,
                        'chain': chain, 
                        'layer': layer, 
                        'type': qa_type,
                        'object_id': object_id,
                        'connection_up': connection_up, 
                        'connection_down': connection_down, 
                        'from': 'human', 
                        'value': question,
                        'object_tags': object_tags})
        
        qa_list.append({'qid': qid,
                        'chain': chain, 
                        'layer': layer, 
                        'type': qa_type,
                        'object_id': object_id,
                        'connection_up': connection_up, 
                        'connection_down': connection_down, 
                        'from': 'rule', 
                        'value': answer,
                        'object_tags': object_tags})


    def get_key_of_key_object(self, key_object_infos, object_dict=None):
        if object_dict is not None:
            keys = [k for k, v in key_object_infos.items() if object_dict['id']==v['id']]
            return keys
        return []
    
    def generate_perception_questions(self, measurements, scenario, scenario_name, route_number, frame_number):
        """
        Generates perception-based questions and answers based on the given scene data, current measurements,
        and scenario. It processes various objects in the scene, such as vehicles, pedestrians, traffic lights,
        stop signs, and landmarks, and generates questions and answers related to these objects.

        Args:
            scene_data (list): List of dictionaries containing information about objects in the scene.
            measurements (dict): Dictionary containing current measurement data.
            scenario (str): The current scenario.

        Returns:
            combined_qas (dict): Dictionary containing lists of question-answer pairs for different categories.
            num_questions (int): Total number of questions generated.
            num_objects (int): Total number of objects in the scene.
            num_questions_per_category (dict): Dictionary containing the number of questions for each category.
            key_object_infos (dict): Dictionary containing information about objects in the scene.
        """

        self.current_measurement_dict = measurements

        self.merging_and_needs_accelerate = None
        self.merging_and_needs_stop = None
        self.merging_danger_vehicles = []
        # if self.in_carla:
        self.scenario_type = measurements.get('scenario_type', None)
        self.trigger_point = measurements.get('trigger_point', None)
        self.other_info = measurements.get('other_info', None)
        self.role_actor_dict = measurements.get('role_actor', None)
        self.role_actor = None
        self.role_transform = None
        self.blocker_actor = None
        self.blocker_transform = None
        print_debug(f"[debug] role_actor_dict is None: {self.role_actor_dict is None}")
        if isinstance(self.role_actor_dict, dict):
            self.role_actor = self.role_actor_dict.get('actor', None)
            self.role_transform = self.role_actor_dict.get('transform', None)
            print_debug(f"[debug] role_actor is None: {self.role_actor is None}")
            print_debug(f"[debug] role_actor_dict is None: {self.role_actor_dict is None}")
        self.blocker_dict = measurements.get('blocker', None)
        if isinstance(self.blocker_dict, dict):
            self.blocker_actor = self.blocker_dict.get('actor', None)
            self.blocker_transform = self.blocker_dict.get('transform', None)
        self.scenario_status = measurements.get('scenario_status', None)
        
        if self.current_measurement_index == 0:
            self.valid_speed_limits = []
            self.last_road_id = INVALID_NUM # random imposible number
            self.last_lane_id = INVALID_NUM
            self.opposite_flag = False
            self.last_special_move_index = INVALID_NUM
            self.last_left_lane = INVALID_NUM
            self.first_lane_command = 4 # follow_lane
            if not self.in_carla:
                self.first_walker_position = find_walker_location(self.current_measurement_path)
            self.vehicle_ids_following_ego = []
            self.role_vehicle_ids = []
            self.answer43_changelane = ""
            self.answer43_brake = ""
            self.crossed_junction_flag = False

            self.merging_and_needs_accelerate = None
            self.merging_and_needs_stop = None
            self.merging_danger_vehicles = []
            self.stopped_at_stop_sign = False
            self.obstacle_blocked_lane_id = None

            self.accelerate_white_list = {}
            self.accelerate_black_list = {}
            self.first_appear_intervals = {}
            self.front_merge_vehicle_ids = {}

        self.side_blind_spot_flag = False
        # for actor_id in self.accelerate_white_list:
        #     if self.accelerate_white_list[actor_id] > 0:
        #         self.accelerate_white_list[actor_id] -= 1
        
        print_debug(f"[debug] frame {self.current_measurement_index}'s acc_whitelist = {self.accelerate_white_list}")
        print_debug(f"[debug] frame {self.current_measurement_index}'s apear_intervals = {self.first_appear_intervals}")
        print_debug(f"[debug] self.front_merge_vehicle_ids = {self.front_merge_vehicle_ids}")

        scene_data = measurements['bounding_boxes']
        
        ego_measurements = {k: v for k, v in measurements.items() if k != 'sensors'}
        # Initialize lists to store different types of objectss
        static_cars = []
        static_objects = []
        other_vehicles = []
        ego_vehicle = None
        pedestrians = []
        traffic_lights = []
        # old_traffic_lights = []
        traffic_signs = []
        speed_limit_signs = []
        stop_signs = []
        # landmarks = []
        # landmark_ids = []   # Needed to avoid duplicates of landmarks
        vehicles_by_id = {}

        self.all_vehicles_info = {}

        # Find ego vehicle first
        for actor in scene_data:
            if actor['class'] == 'ego_vehicle':
                ego_vehicle = actor
                ego = actor

        # Preprocess, add some keys
        ego_location = carla.Location(x=ego['location'][0], y=ego['location'][1], z=ego['location'][2])
        lane_info = get_lane_info(self.map, ego_location)
        for key, value in lane_info.items():
            if key not in ego:
                ego[key] = value

        normal_speed_vehicle_count = 0
        normal_speed_sum = 0.0

        for actor in scene_data:
            # relative position
            actor['id'] = str(actor['id'])
            actor['position'] = transform_to_ego_coordinates(actor['location'], ego['world2ego'])
            actor['yaw'] = math.radians(actor["rotation"][2] - ego_vehicle["rotation"][2])
            if actor['class'] == 'vehicle':
                if actor['speed'] >= SLOW_VEHICLE_SPEED and actor['distance'] < SURROUNDING_SPEED_RADIUS:
                    normal_speed_vehicle_count += 1
                    normal_speed_sum += actor['speed']
                other_vehicles_info = get_other_vehicle_info(self.current_measurement_path, self.map, actor, ego)
                for key, value in other_vehicles_info.items():
                    if key not in actor:
                        actor[key] = value
                if 'base_type' not in actor:
                    actor['base_type'] = 'vehicle'
                actor['relative_speed'], actor['approaching_dot_product'] = compute_relative_velocity(ego_vehicle, actor)
                if actor['approaching_dot_product'] > 0:
                    tmp_value = self.accelerate_black_list.pop(actor['id'], None)
                    tmp_value = self.accelerate_white_list.pop(actor['id'], None)
                if actor['id'] in self.accelerate_white_list:
                    self.accelerate_white_list[actor['id']] = actor['distance']
                if actor['id'] in self.accelerate_black_list:
                    self.accelerate_black_list[actor['id']] = actor['distance']
                actor['intersection_point'], actor['ego_to_intersection'], actor['actor_to_intersection'] = compute_intersection_distance(actor, ego_vehicle)
                actor['actor_arrive_intersection_time'] = actor['actor_to_intersection'] / actor['speed'] if actor['actor_to_intersection'] is not None else None
                actor['intersection_point_left'], actor['ego_to_intersection_left'], actor['actor_to_intersection_left'] = compute_intersection_distance(actor, ego_vehicle, -TURN_DEVIATE_DEGREE)
                actor['intersection_point_right'], actor['ego_to_intersection_right'], actor['actor_to_intersection_right'] = compute_intersection_distance(actor, ego_vehicle, TURN_DEVIATE_DEGREE)
                actor['actor_arrive_intersection_time_left'] = actor['actor_to_intersection_left'] / actor['speed'] if actor['actor_to_intersection_left'] is not None else None
                actor['actor_arrive_intersection_time_right'] = actor['actor_to_intersection_right'] / actor['speed'] if actor['actor_to_intersection_right'] is not None else None
                actor['lane_yaw_degree'], actor['vehicle_yaw_degree'], actor['lane_deviant_degree'] = get_lane_deviate_info(self.map, actor)
                if not self.in_carla:
                    actor['steer'] = get_steer_by_future(self.current_measurement_path, actor['id'])
                    actor['vehicle_cuts_in'] = is_vehicle_cutting_in(ego, actor, self.map, self.current_measurement_path)
                else:
                    actor['vehicle_cuts_in'] = False
                    if (not ego['is_in_junction']):
                        if self.scenario_status is not None and self.role_actor is not None and 'CutIn' in self.scenario_type and 'Highway' not in self.scenario_type:
                            if self.other_info['direction']['value'] == 'right':
                                actor['vehicle_cuts_in'] = (actor['id'] == str(self.role_actor.id) and \
                                                            5.0 > actor['position'][1] > -1.0 and \
                                                            actor['position'][0] > BACK_CONSIDER_THRESHOLD and \
                                                            actor['distance'] < CHANGE_LANE_THRESHOLD)
                                if actor['vehicle_cuts_in']:
                                    actor['vehicle_cuts_in'] = bool(abs(actor['wp_angle']) > 10.0)
                            if self.other_info['direction']['value'] == 'left':
                                actor['vehicle_cuts_in'] = (actor['id'] == str(self.role_actor.id) and \
                                                            -5.0 < actor['position'][1] < 1.0 and \
                                                            actor['position'][0] > BACK_CONSIDER_THRESHOLD and \
                                                            actor['distance'] < CHANGE_LANE_THRESHOLD)
                                if actor['vehicle_cuts_in']:
                                    actor['vehicle_cuts_in'] = bool(abs(actor['wp_angle']) > CUT_IN_DEVIATION)
                        actor['vehicle_cuts_in'] = bool(actor['vehicle_cuts_in'])
                        
                        if 'lane_relative_to_ego' not in actor or actor['lane_relative_to_ego'] is None:
                            actor['lane_relative_to_ego'] = INVALID_NUM
                        if actor['position'][0] > 0.0 and actor['distance'] < CHANGE_LANE_THRESHOLD and actor['speed'] > 1.0:
                            if actor['lane_deviant_degree'] > CUT_IN_DEVIATION and 0 <= actor['lane_relative_to_ego'] <= 1:
                                actor['vehicle_cuts_in'] = True
                            elif actor['lane_deviant_degree'] < -CUT_IN_DEVIATION and 0 >= actor['lane_relative_to_ego'] >= -1:
                                actor['vehicle_cuts_in'] = True
        # Categorize objects from the scene data
        # print(scene_data) # debug
                
        for actor in scene_data:
            if actor['class'] == 'vehicle':
                other_vehicles.append(actor)
                vehicles_by_id[actor['id']] = actor
                if actor['state'] == 'static':
                    static_cars.append(actor)
            elif actor['class'] == 'walker': 
                # actually, b2d don't have walker either.
                pedestrians.append(actor)
            # elif actor['class'] == 'landmark' and actor['id'] not in landmark_ids:
            #     landmarks.append(actor)
            #     landmark_ids.append(actor['id'])
            # elif actor['class'] == 'traffic_light_vqa':
            #     traffic_lights.append(actor)
            elif actor['class'] == 'traffic_light':
                # print_debug(f"[debug] traffic light id={actor['id']}, affects_ego={actor['affects_ego']}")
                if self.in_carla and not is_vehicle_in_camera(self.CAMERA_FRONT, actor):
                    actor['affects_ego'] = False # sometimes some traffic light bug occur in carla...
                traffic_lights.append(actor)
            elif actor['class'] == 'traffic_sign' or 'stop' in actor['type_id']: # stop maybe on the ground
                traffic_signs.append(actor)
                if 'stop' in actor['type_id']:
                    stop_signs.append(actor)
                if 'static' in actor['type_id']:
                    static_objects.append(actor)
                if 'speed_limit' in actor['type_id']:
                    speed_limit_signs.append(actor)
            elif 'static' in actor['class']:
                static_objects.append(actor)

        important_objects = []
        key_object_infos = {}
        
        # _, ego['distance_to_junction'] = find_first_junction_in_direction(self.map, ego_location)
        # ego['is_in_junction'] = is_vehicle_in_junction(self.map, ego_location)
        print_debug(f"[debug] self.accelerate_black_list = {self.accelerate_black_list}")

        self.ideal_flow_speed = NORMAL_KEEP_SPEED
        if scenario_name in self.enter_highway_scenarios or scenario_name in self.leave_highway_scenarios:
            self.ideal_flow_speed = HIGHWAY_KEEP_SPEED
        if normal_speed_vehicle_count > 0:
            self.ideal_flow_speed = normal_speed_sum / normal_speed_vehicle_count
        measurements['ideal_flow_speed'] = self.ideal_flow_speed

        ego['(debug) ego_location'] = transform_to_ego_coordinates(ego['location'], ego['world2ego'])

        ego['lane_yaw_degree'], ego['ego_yaw_degree'], ego['lane_deviant_degree'] = get_lane_deviate_info(self.map, ego)

        if not self.in_carla:
            # ego['virtual_steer'] = measurements['steer'] # get_steer_by_future(self.current_measurement_path, ego['id'])
            ego['steer'] = measurements['steer']
        # else:
        #     ego['virtual_steer'] = ego['steer']
        ego['hazard_detected_10'] = False
        res = vehicle_obstacle_detected(ego, other_vehicles, self.map, 10)
        affected_by_vehicle_10, hazard_actor_10 = res
        if affected_by_vehicle_10:
                ego['hazard_detected_10'] = True
                ego['affects_ego_10'] = hazard_actor_10['id']
                ego['affects_ego_10_id'] = hazard_actor_10['id']
                ego['affects_ego_10_dis'] = hazard_actor_10['distance']

        ego['hazard_detected_15'] = False
        affected_by_vehicle_15, hazard_actor_15 = vehicle_obstacle_detected(ego, other_vehicles, self.map, 15)
        if affected_by_vehicle_15:
                ego['hazard_detected_15'] = True
                ego['affects_ego_15'] = hazard_actor_15['id']
                ego['affects_ego_15_id'] = hazard_actor_15['id']
                ego['affects_ego_15_dis'] = hazard_actor_15['distance']

        ego['hazard_detected_20'] = False
        affected_by_vehicle_20, hazard_actor_20 = vehicle_obstacle_detected(ego, other_vehicles, self.map, 20)
        if affected_by_vehicle_20:
                ego['hazard_detected_20'] = True
                ego['affects_ego_20'] = hazard_actor_20['id']
                ego['affects_ego_20_id'] = hazard_actor_20['id']
                ego['affects_ego_20_dis'] = hazard_actor_20['distance']        
        ego['hazard_detected_40'] = False
        affected_by_vehicle_40, hazard_actor_40 = vehicle_obstacle_detected(ego, other_vehicles, self.map, 40)
        if affected_by_vehicle_40:
                ego['hazard_detected_40'] = True
                ego['affects_ego_40'] = hazard_actor_40['id']
                ego['affects_ego_40_id'] = hazard_actor_40['id']
                ego['affects_ego_40_dis'] = hazard_actor_40['distance']

        # original only raise this flag when ego vehicle overcomes an obstacle
        # measurements['changed_route'] = is_ego_changing_lane_due_to_obstacle(ego_measurements, self.map, scene_data)
        # _, _, measurements['changed_route'], _, _ = detect_lane_change_by_time(self.map, ego['id'], ego, self.current_measurement_path)
        # measurements['control_brake'] = measurements['should_brake']
        measurements['command'] = measurements['command_near']
        measurements['target_point'] = [measurements['x_target'], measurements['y_target']]

        # if self.in_carla:
        #     # print(f"[debug] measurements['x_command_near'] = {measurements['x_command_near']}, measurements['y_command_near'] = {measurements['y_command_near']}")
        #     # coordinates are relative
        #     relative_command_far = [measurements['x_command_far'], measurements['y_command_far'], ego['location'][2]]
        #     relative_command_near = [measurements['x_command_near'], measurements['y_command_near'], ego['location'][2]]
        #     absolute_command_far = transform_to_world_coordinates(relative_command_far, ego['world2ego'])
        #     absolute_command_near = transform_to_world_coordinates(relative_command_near, ego['world2ego'])
        #     measurements['x_command_far'], measurements['y_command_far'] = absolute_command_far[0], absolute_command_far[1]
        #     measurements['x_command_near'], measurements['y_command_near'] = absolute_command_near[0], absolute_command_near[1]
        #     measurements['x_target'], measurements['y_target'] = measurements['x_command_far'], measurements['y_command_far']
        #     measurements['target_point'] = [measurements['x_target'], measurements['y_target']]

        # change lane calculations
        if self.first_lane_command == 4:
            if measurements['command_near'] in [5, 6]:
                self.first_lane_command = measurements['command_near']
            if measurements['command_far'] in [5, 6]:
                self.first_lane_command = measurements['command_far']
        
        # target lane calculations
        self.correct_road = None
        self.correct_lane = None
        if not ego['is_in_junction']:
            if measurements['command_near'] == 5: # change left lane
                self.correct_road, self.correct_lane = get_relative_lane_id(self.map,
                                                                            x=measurements['x_command_near'],
                                                                            y=measurements['y_command_near'],
                                                                            z=ego['location'][2],
                                                                            n=-1)
            elif measurements['command_near'] == 6: # change right lane
                self.correct_road, self.correct_lane = get_relative_lane_id(self.map,
                                                                            x=measurements['x_command_near'],
                                                                            y=measurements['y_command_near'],
                                                                            z=ego['location'][2],
                                                                            n=1)
            else: # follow lane, turn left, turn right
                self.correct_road, self.correct_lane = get_relative_lane_id(self.map,
                                                                            x=measurements['x_command_near'],
                                                                            y=measurements['y_command_near'],
                                                                            z=ego['location'][2],
                                                                            n=0)
                
        measurements['correct_road'] = self.correct_road
        measurements['correct_lane'] = self.correct_lane

        ego['angle_command_near'] = world_point_deviate_info(target_x=measurements['x_command_near'],
                                                             target_y=measurements['y_command_near'],
                                                             vehicle=ego_vehicle)
        ego['angle_command_far'] = world_point_deviate_info(target_x=measurements['x_command_far'],
                                                            target_y=measurements['y_command_far'],
                                                            vehicle=ego_vehicle)
        ego['angle_exit'] = None
        if measurements['junction_exit_wp_x'] is not None and measurements['junction_exit_wp_y'] is not None:
            ego['angle_exit'] = world_point_deviate_info(target_x=measurements['junction_exit_wp_x'],
                                                        target_y=measurements['junction_exit_wp_y'],
                                                        vehicle=ego_vehicle)

        # speed limit calculations
        for sign in speed_limit_signs:
            existed = False
            for existed_sign in self.valid_speed_limits:
                if existed_sign['id'] == sign['id']:
                    affect_flag = existed_sign['affects_ego'] or sign['affects_ego']
                    for key, value in sign.items():
                        existed_sign[key] = value
                    existed_sign['affects_ego'] = affect_flag
                    existed = True
            if existed is False and sign['affects_ego'] is True and sign['distance'] < SPEED_LIMIT_CONSIDER_RAIUS:
                self.valid_speed_limits.append(sign)
        
        new_road_id = ego_vehicle['road_id']
        new_lane_id = ego_vehicle['lane_id']
        # if self.last_lane_id * new_lane_id < 0:
        #     if self.last_road_id == new_road_id:
        #         self.opposite_flag = not self.opposite_flag
        
        self.opposite_flag = False
        if abs(ego['lane_deviant_degree']) > OPPOSITE_ANGLE_THRESHOLD and ego_vehicle['num_lanes_opposite_direction'] > 0:
            self.opposite_flag = True

        if self.last_road_id != new_road_id:
            self.valid_speed_limits = [x for x in self.valid_speed_limits if x['position'][0] >= SPEED_LIMIT_VALID_THRESHOLD]
            # the position threshold is loose because of HighwayExit_Town06_Route291_Weather5
        self.last_road_id = new_road_id
        self.last_lane_id = new_lane_id

        self.crossed_junction_flag = self.crossed_junction_flag or ego_vehicle['is_in_junction']
        
        sorted_speed_limits = sorted(self.valid_speed_limits, key=lambda x: x['distance'])

        self.ahead_speed_limit = next(
            (item for item in sorted_speed_limits if item['position'][0] > SPEED_LIMIT_AHEAD_THRESHOLD and item['affects_ego']),
            None
        )

        self.passed_speed_limit = next(
            (item for item in sorted_speed_limits if item['position'][0] <= SPEED_LIMIT_AHEAD_THRESHOLD),
            None
        )

        self.current_speed_limit = INF_MAX
        if self.passed_speed_limit is not None:
            self.current_speed_limit = int(self.passed_speed_limit['type_id'].split('.')[-1])

        self.future_speed_limit = INF_MAX
        if self.ahead_speed_limit is not None:
            self.future_speed_limit = int(self.ahead_speed_limit['type_id'].split('.')[-1])

        for vehicle in vehicles_by_id.values():
            if vehicle['lane_relative_to_ego'] is not None and vehicle['lane_relative_to_ego'] == 0:
                if vehicle['position'][0] < 0.0:
                    self.vehicle_ids_following_ego.append(vehicle['id'])
        
        right_front_clear = True
        left_front_clear = True

        if (not self.opposite_flag and SIDE_FRONT_CLEAR_ANGLE < ego['lane_deviant_degree'] < 180) or \
           (self.opposite_flag and SIDE_FRONT_CLEAR_ANGLE - 180 < ego['lane_deviant_degree'] < 0):
            # deviate too much to the left
            right_front_clear = False
        if (not self.opposite_flag and -SIDE_FRONT_CLEAR_ANGLE > ego['lane_deviant_degree'] > -180) or \
           (self.opposite_flag and -SIDE_FRONT_CLEAR_ANGLE + 180 > ego['lane_deviant_degree'] > 0):
            # deviate too much to the right
            left_front_clear = False
        
        for actor in scene_data:
            if actor['class'] == 'vehicle' and 'state' in actor and actor['state'] != 'dynamic':
                continue
            if 'type_id' in actor and 'traffic' in actor['type_id']:
                continue # ignore traffic signs
            actor_pos = transform_to_ego_coordinates(actor['location'], ego['world2ego'])
            if SIDE_FRONT_CLEAR_X_MIN < actor_pos[0] < SIDE_FRONT_CLEAR_X_MAX and \
               SIDE_FRONT_CLEAR_Y_MIN < actor_pos[1] < SIDE_FRONT_CLEAR_Y_MAX:
                print_debug(f"[debug] actor id = {actor['id']} blocks the ego vehicle in front right.")
                right_front_clear = False
            if right_front_clear == False:
                break
        for actor in scene_data:
            if actor['class'] == 'vehicle' and 'state' in actor and actor['state'] != 'dynamic':
                continue
            if 'type_id' in actor and 'traffic' in actor['type_id']:
                continue # ignore traffic signs
            actor_pos = transform_to_ego_coordinates(actor['location'], ego['world2ego'])
            if SIDE_FRONT_CLEAR_X_MIN < actor_pos[0] < SIDE_FRONT_CLEAR_X_MAX and \
                -SIDE_FRONT_CLEAR_Y_MIN > actor_pos[1] > -SIDE_FRONT_CLEAR_Y_MAX:
                print_debug(f"[debug] actor id = {actor['id']} blocks the ego vehicle in front left.")
                left_front_clear = False
            if left_front_clear == False:
                break
        
        opposite_and_right_front_clear = self.opposite_flag and right_front_clear
        ego['opposite_and_right_front_clear'] = opposite_and_right_front_clear
        ego['right_front_clear'] = right_front_clear
        ego['left_front_clear'] = left_front_clear

        if 'distance_to_junction' not in ego or ego['distance_to_junction'] is None:
             ego['distance_to_junction'] = INF_MAX

        # Generate questions and answers for different categories
        res = generate_vehicle_information(self, other_vehicles, ego, important_objects, key_object_infos,
                                           scene_data, vehicles_by_id, measurements, scenario)

        qas_conversation_vehicle, important_objects, key_object_infos = res

        res = process_traffic_signs(self, traffic_signs, important_objects, key_object_infos)
        qas_conversation_trafficsign, important_objects, key_object_infos, ts_info, ts_object_tags = res

        res = process_traffic_lights(self, traffic_lights, ego, important_objects, key_object_infos)
        qas_conversation_trafficlight, important_objects, key_object_infos, tl_info, tl_object_tags = res
        
        res = process_pedestrians(self, pedestrians, important_objects, key_object_infos)
        qas_conversation_pedestrian, important_objects, key_object_infos = res
        
        res = generate_ego_vehicle_actions(self, ego_vehicle, pedestrians, ego, important_objects, key_object_infos,
                                                vehicles_by_id, tl_info, ts_info, static_objects, measurements, scene_data,
                                                scenario, stop_signs, ts_object_tags, tl_object_tags)
        qas_conversation_ego, important_objects, key_object_infos, \
            final_change_dir, final_lane_change_flag, \
            changed_for_real, final_brake_flag, final_stop_flag = res

        res = analyze_road_layout(self, ego, other_vehicles, scene_data, important_objects, key_object_infos, measurements, scenario, final_change_dir)
        qas_conversation_roadlayout, important_objects, key_object_infos = res

        res = analyze_environment(self, ego, other_vehicles, scene_data, measurements, scenario)
        qas_conversation_environment = res

        res = answer_behaviour_questions(self, ego, other_vehicles, scene_data, measurements, scenario,
                                              important_objects, key_object_infos,
                                              final_lane_change_flag=final_lane_change_flag,
                                              final_change_dir=final_change_dir,
                                              changed_for_real=changed_for_real,
                                              final_brake_flag=final_brake_flag,
                                              final_stop_flag=final_stop_flag)
        qas_conversation_behaviour, final_dir_cmd, final_spd_cmd, waiting_for_red_light, is_trivial_case = res

        ego['not_passed_circumvent_obstacles'] = self.not_passed_circumvent_obstacles
        ego['distance_to_circumvent_obstacle'] = self.distance_to_circumvent_obstacle
        
        ######## for [debug] appendix ########
        final_ego_data = filter_serializable(ego)
        self.appended_measurements = filter_serializable(measurements)
        final_scene_data = filter_serializable(scene_data)

        append_dict = {
            'ego': final_ego_data,
            'measurements': self.appended_measurements,
            'scene_data': final_scene_data
        }

        file_name = f'{self.output_graph_directory}/appendix/' \
                            f'{self.save_name}/{int(frame_number):05d}.json'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(append_dict, f, sort_keys=True, indent=4)
        ######## for [debug] appendix ########

        num_objects = len(important_objects)
        num_questions = len(qas_conversation_vehicle) + len(qas_conversation_roadlayout) + \
                        len(qas_conversation_trafficsign) + len(qas_conversation_trafficlight) + \
                        len(qas_conversation_ego) + len(qas_conversation_environment) + \
                        len(qas_conversation_behaviour)
        num_questions = num_questions // 2 # Because we have two entries per question
        num_questions += 1 # Because we have the question about the important objects

        num_questions_per_category = {
            'dynamic_vehicles': len(qas_conversation_vehicle) // 2,
            'roadlayout': len(qas_conversation_roadlayout) // 2,
            'trafficsign': len(qas_conversation_trafficsign) // 2,
            'trafficlight': len(qas_conversation_trafficlight) // 2,
            'pedestrian': len(qas_conversation_pedestrian) // 2,
            'environment': len(qas_conversation_environment) // 2,
            'behaviour': len(qas_conversation_behaviour) // 2,
            'ego': len(qas_conversation_ego) // 2,
        }

        qas_conversation_objects = []
        question = 'What are the important objects in the scene?'
        concatenated_important_objects = ''

        # Merge same objects and count identical objects in the same direction
        grouped_items = {}
        keep_items = []
        for obj_idx, obj in enumerate(important_objects):
            item_parts = obj.split(" to the ")
            if item_parts[0].startswith('the '):
                item_parts[0] = item_parts[0][4:]
            if len(item_parts) == 1:
                keep_items.append(obj)
            else:
                if item_parts[1] not in grouped_items:
                    grouped_items[item_parts[1]] = []
                grouped_items[item_parts[1]].append(item_parts[0])

        result = []
        for key, values in grouped_items.items():
            counted_values = dict(Counter(values))
            organize = []
            for key1, values1 in counted_values.items():
                if values1 > 1:
                    organize.append((f'the {values1} {key1}s'))
                else:
                    organize.append((f'the {key1}'))

            res = ''
            for obj_idx, obj in enumerate(organize):
                separator = ', '
                if obj_idx+1 == len(organize)-1:
                    separator = ' and '
                if obj_idx == len(organize)-1:
                    separator = ''
                res += f'{obj}{separator}'
            result.append(res+ f' to the {key}')

        # Merge result with keep_items
        important_objects_merged = keep_items + result

        # Concatenate important objects for the answer
        for obj_idx, obj in enumerate(important_objects_merged):
            separator = ','
            if obj_idx+1 == len(important_objects_merged)-1:
                separator = ' and'
            if obj_idx == len(important_objects_merged)-1:
                separator = ''
            concatenated_important_objects += f' {obj}{separator}'

        if len(important_objects_merged) == 0:
            answer = 'There is no important object in the scene.'
        elif len(important_objects) == 1:
            answer = f'The important object in the scene is{concatenated_important_objects}.'
        else:
            answer = f'The important objects in the scene are{concatenated_important_objects}.'

        # Add the question and answer to the conversation
        self.add_qas_questions(qa_list=qas_conversation_objects, 
                                qid=18,
                                chain=0,
                                layer=0,
                                qa_type='perception',
                                connection_up=-1,
                                connection_down=-1,
                                question=question,
                                answer=answer,
                                object_tags=list(key_object_infos.keys()))
        
        # question about importancy.
        order_question = "What are the important objects in the scene? " + \
                            "List them from most to least important."
        
        for key, key_dict in key_object_infos.items():
            same_dir = False
            is_sign = False
            if key_dict['obj_dict'] is not None:
                road_id = key_dict['obj_dict'].get('road_id', INVALID_NUM)
                lane_id = key_dict['obj_dict'].get('lane_id', INVALID_NUM)
                same_dir = (ego_vehicle['road_id'] == road_id and ego_vehicle['lane_id'] * lane_id >= 0)
                class_name = key_dict['obj_dict'].get('class', 'None')
                is_sign = class_name == 'traffic_sign' or class_name == 'traffic_light'
            key_dict['same_dir'] = same_dir
            key_dict['is_sign'] = is_sign

        sorted_items = sorted(
            key_object_infos.items(),
            key=lambda item: (not item[1]['is_role'], not item[1]['is_sign'], not item[1]['is_dangerous'], 
                              not item[1]['same_dir'], item[1]['distance'])
        )
        sorted_keys = [item[0] for item in sorted_items]
        sorted_values = [item[1] for item in sorted_items]
        
        if len(sorted_values) > 0:
            order_answer = f"{sorted_values[0]['Detailed_description']}({sorted_keys[0]})"
            for i in range(1, len(sorted_values)):
                order_answer = f"{order_answer}, {sorted_values[i]['Detailed_description']}({sorted_keys[i]})"

            order_answer = f"{order_answer[0].upper()}{order_answer[1:]}."
        else:
            order_answer = 'There is no important object in the scene.'
        
        self.add_qas_questions(qa_list=qas_conversation_objects, 
                            qid=19,
                            chain=0,
                            layer=0,
                            qa_type='perception',
                            connection_up=-1,
                            connection_down=-1,
                            question=order_question,
                            answer=order_answer,
                            object_tags=sorted_keys)

        for key, value in self.all_vehicles_info.items():
            is_role = any(info['id'] == key and info['is_role'] for info in key_object_infos.values())
            value['is_role'] = is_role

        self.all_vehicles_info_list = sorted(
            self.all_vehicles_info.items(),
            key=lambda item: (not item[1]['is_role'], item[1]['distance'])
        )

        important_vehicles = []
        for item in self.all_vehicles_info_list:
            key, value = item[0], item[1]
            if value['consider'] is True:
                important_vehicles.append(value)
        other_vehicle_count = len(important_vehicles)

        if True:
            list_condition = "The list of important vehicles in the current scene is '"
            for i in range(other_vehicle_count):
                if i == other_vehicle_count - 1:
                    list_condition = f"{list_condition}the {important_vehicles[i]['description']}.' "
                else:
                    list_condition = f"{list_condition}the {important_vehicles[i]['description']}, "
            if other_vehicle_count == 0:
                list_condition = "There's no important vehicle in the current scene. "

            question1 = f"{list_condition}What are the rough moving speed and moving direction of them?"
            answer1 = ""
            question2 = f"{list_condition}What are the exact moving speed and moving direction of them?"
            answer2 = ""
            question5 = f"{list_condition}Where on the road are they located?"
            answer5 = ""
            question6 = f"What are the important vehicles in the scene and where on the road are they located? " +\
                        "List them from most to least important."
            answer6 = ""

            obj_tags = []
            obj_ids = []
            for i in range(other_vehicle_count):
                vdict = important_vehicles[i]
                obj_ids.extend(vdict["obj_id"])
                obj_tags.extend(vdict["obj_tags"])

                single_question = f"Where on the road is the {vdict['description']} located?"
                single_answer = f"The {vdict['description']} {vdict['position_str']}."

                self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=20,
                                        chain=4,
                                        layer=0,
                                        qa_type='perception',
                                        connection_up=[(4,1), (4,2), (4,3)],
                                        connection_down=[(3,0),(3,2),(3,3)],
                                        question=single_question,
                                        answer=single_answer,
                                        object_id=vdict["obj_id"],
                                        object_tags=vdict["obj_tags"])

                single_question = f"What is the rough moving speed and moving direction of {vdict['description']}?"
                single_answer = f"The {vdict['description']} is {vdict['motion']}, {vdict['dir']}."

                self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=21,
                                        chain=4,
                                        layer=1,
                                        qa_type='prediction',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=single_question,
                                        answer=single_answer,
                                        object_id=vdict["obj_id"],
                                        object_tags=vdict["obj_tags"])
                
                single_question = f"What is the exact moving speed and moving direction of {vdict['description']}?"
                single_answer = f"The {vdict['description']} is driving at the speed of {vdict['speed']:.1f} m/s, {vdict['dir']}."

                self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=22,
                                        chain=4,
                                        layer=1,
                                        qa_type='prediction',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=single_question,
                                        answer=single_answer,
                                        object_id=vdict["obj_id"],
                                        object_tags=vdict["obj_tags"])
                
                single_question = f"The ego vehicle {self.ego_command_str}. " +\
                    f"Is {vdict['description']} potentially crossing the path of the ego vehicle? If so, why?"
                single_question2 = f"The ego vehicle {self.ego_command_str}. " +\
                    f"Is {vdict['description']} potentially crossing the path of the ego vehicle? If so, why?" +\
                        " And what action can lead to a collision?"
                single_question3 = f"The ego vehicle {self.ego_command_str}. " +\
                    f"Is {vdict['description']} potentially crossing the path of the ego vehicle? If so," +\
                        " what action can lead to a collision?"
                
                if vdict['cross_flag'] is True:
                    single_answer = f"Yes, the {vdict['description']} {vdict['cross_reason']}, so the ego vehicle " +\
                        "should pay attention to not crash into it."
                    single_answer2 = f"Yes, the {vdict['description']} {vdict['cross_reason']}, and the collision " +\
                        f"will happen if the ego vehicle {vdict['cross_action']}."
                    single_answer3 = f"Yes, the collision " +\
                        f"will happen if the ego vehicle {vdict['cross_action']}."
                else:
                    single_answer = f"No, the {vdict['description']} is not crossing paths with the ego vehicle."
                    single_answer2 = f"No, the {vdict['description']} is not crossing paths with the ego vehicle."
                    single_answer3 = f"No, the {vdict['description']} is not crossing paths with the ego vehicle."
                
                self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=23,
                                        chain=4,
                                        layer=1,
                                        qa_type='planning',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=single_question,
                                        answer=single_answer,
                                        object_id=vdict["obj_id"],
                                        object_tags=vdict["obj_tags"])
                
                self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=48,
                                        chain=4,
                                        layer=1,
                                        qa_type='planning',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=single_question2,
                                        answer=single_answer2,
                                        object_id=vdict["obj_id"],
                                        object_tags=vdict["obj_tags"])
                
                self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=49,
                                        chain=4,
                                        layer=1,
                                        qa_type='planning',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=single_question3,
                                        answer=single_answer3,
                                        object_id=vdict["obj_id"],
                                        object_tags=vdict["obj_tags"])

                answer1 = f"{answer1} The {vdict['description']} is {vdict['motion']}, {vdict['dir']}."
                answer2 = f"{answer2} The {vdict['description']} is driving at the speed of {vdict['speed']:.1f} m/s, {vdict['dir']}."
                answer5 = f"{answer5} The {vdict['description']} {vdict['position_str']}."
                answer6 = f"{answer6} The {vdict['description']}, which {vdict['position_str']}."

            if other_vehicle_count == 0:
                answer1 = "There's no important vehicle in the current scene."
                answer2 = "There's no important vehicle in the current scene."
                answer5 = "There's no important vehicle in the current scene."
                answer6 = "There's no important vehicle in the current scene."
            
            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=24,
                                        chain=4,
                                        layer=1,
                                        qa_type='prediction',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=question1,
                                        answer=answer1,
                                        object_id=obj_ids,
                                        object_tags=obj_tags)
        
            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=25,
                                        chain=4,
                                        layer=1,
                                        qa_type='prediction',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=question2,
                                        answer=answer2,
                                        object_id=obj_ids,
                                        object_tags=obj_tags)
            
            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=26,
                                        chain=4,
                                        layer=1,
                                        qa_type='perception',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=question6,
                                        answer=answer6,
                                        object_id=obj_ids,
                                        object_tags=obj_tags)
            
            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=27,
                                        chain=4,
                                        layer=1,
                                        qa_type='perception',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=question5,
                                        answer=answer5,
                                        object_id=obj_ids,
                                        object_tags=obj_tags)
            
            question3 = f"The ego vehicle {self.ego_command_str}. {list_condition}Among them, please identify those that may "\
                        + "overlap with the ego vehicle's path and provide reasons for the overlap."
            answer3 = ""

            question4 = f"The ego vehicle {self.ego_command_str}. {list_condition}Among them, please identify those that may "\
                        + "overlap with the ego vehicle's path."
            answer4 = ""

            question5 = f"The ego vehicle {self.ego_command_str}. {list_condition}Among them, identify those that may overlap " +\
                         "with the ego vehicle's path and the actions that could lead to a collision."
            answer5 = ""

            question6 = f"The ego vehicle {self.ego_command_str}. {list_condition}Among them, identify those that may overlap " +\
                         "with the ego vehicle's path, provide reasons for the overlap and the actions that could lead to a collision."
            answer6 = ""

            obj_tags = []
            obj_ids = []
            for i in range(other_vehicle_count):
                vdict = important_vehicles[i]
                if vdict['cross_flag'] is False:
                    continue
                
                obj_ids.extend(vdict["obj_id"])
                obj_tags.extend(vdict["obj_tags"])
                answer3 = f"{answer3} The {vdict['description']}, because it {vdict['cross_reason']}."
                answer5 = f"{answer5} The {vdict['description']}, the collision will happen if the ego vehicle {vdict['cross_action']}."
                answer6 = f"{answer6} The {vdict['description']}, because it {vdict['cross_reason']}, and the collision will happen if the ego vehicle {vdict['cross_action']}."
                if answer4 == "":
                    answer4 = f"{answer4}The {vdict['description']}"
                else:
                    answer4 = f"{answer4}, the {vdict['description']}"

            if answer3 == "":
                answer3 = "None of them potentially crossing the path of ego vehicle now."
                answer4 = "None of them potentially crossing the path of ego vehicle now."
                answer5 = "None of them potentially crossing the path of ego vehicle now."
                answer6 = "None of them potentially crossing the path of ego vehicle now."
            else:
                answer4 = f"{answer4}."
            
            if other_vehicle_count == 0:
                answer3 = "There's no important vehicle, so none of them potentially crossing the path of ego vehicle now."
                answer4 = "There's no important vehicle, so none of them potentially crossing the path of ego vehicle now."
                answer5 = "There's no important vehicle, so none of them potentially crossing the path of ego vehicle now."
                answer6 = "There's no important vehicle, so none of them potentially crossing the path of ego vehicle now."

            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=28,
                                        chain=4,
                                        layer=1,
                                        qa_type='planning',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=question3,
                                        answer=answer3,
                                        object_id=obj_ids,
                                        object_tags=obj_tags)
            
            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=29,
                                        chain=4,
                                        layer=1,
                                        qa_type='planning',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=question4,
                                        answer=answer4,
                                        object_id=obj_ids,
                                        object_tags=obj_tags)
            
            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=46,
                                        chain=4,
                                        layer=1,
                                        qa_type='planning',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=question5,
                                        answer=answer5,
                                        object_id=obj_ids,
                                        object_tags=obj_tags)
            
            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                        qid=47,
                                        chain=4,
                                        layer=1,
                                        qa_type='planning',
                                        connection_up=[(4,3)],
                                        connection_down=[(4,0)],
                                        question=question6,
                                        answer=answer6,
                                        object_id=obj_ids,
                                        object_tags=obj_tags)

        combined_qas = {
            'important_objects': qas_conversation_objects,
            'dynamic_vehicles': qas_conversation_vehicle,
            'roadlayout': qas_conversation_roadlayout,
            'trafficsign': qas_conversation_trafficsign,
            'trafficlight': qas_conversation_trafficlight,
            'pedestrian': qas_conversation_pedestrian,
            'environment': qas_conversation_environment,
            'behaviour': qas_conversation_behaviour,
            'ego': qas_conversation_ego,
        }

        # if 'InvadingTurn' in scenario_name:
        #     final_lane_change_flag = False # just deviate, not changing lane.
        all_spd_cmd = SpeedCommand()
        all_dir_cmd = DirectionCommand()
        final_lane_change_flag = final_dir_cmd in [all_dir_cmd.left_change, all_dir_cmd.right_change]
        if self.in_carla:
            real_speed_change = "Ambiguous"
        else:
            real_speed_change = get_acceleration_by_future(self.current_measurement_path, k=10)
        
        extra_flags = {
            'ego_speed': ego_vehicle['speed'],
            'change_lane_flag': final_lane_change_flag,
            'change_lane_gt': changed_for_real,
            'speed_cmd': final_spd_cmd,
            'direction_cmd': final_dir_cmd,
            'speed_change_gt': real_speed_change,
            'waiting_for_red_light': waiting_for_red_light, 
            'is_trivial_case': is_trivial_case
        }

        return combined_qas, num_questions, num_objects, num_questions_per_category, key_object_infos, extra_flags