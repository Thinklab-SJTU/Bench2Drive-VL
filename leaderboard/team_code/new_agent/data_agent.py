"""
Child of the autopilot that additionally runs data collection and storage.
"""

import cv2
import carla
import random
import torch
import numpy as np
import json
import os
import gzip
import laspy
import webcolors
from shapely.geometry import Polygon
from pathlib import Path

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.envs.sensor_interface import SensorInterface, CallBack, OpenDriveMapReader, SpeedometerReader
import xml.etree.ElementTree as ET
from easydict import EasyDict
import math
import h5py
from utils import build_projection_matrix, convert_depth, get_relative_transform, normalize_angle, build_skeleton,  get_matrix, calculate_cube_vertices, compute_2d_distance
from utils import DIS_CAR_SAVE, DIS_WALKER_SAVE, DIS_SIGN_SAVE, DIS_LIGHT_SAVE


from autopilot import AutoPilot
import transfuser_utils as t_u

from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign
from PIL import Image

from agents.tools.misc import (get_speed, is_within_distance,
                                 get_trafficlight_trigger_location,
                                 compute_distance)

from agents.navigation.local_planner import LocalPlanner, RoadOption


# from: https://medium.com/codex/rgb-to-color-names-in-python-the-robust-way-ec4a9d97a01f
from scipy.spatial import KDTree
from webcolors import (
    CSS2_HEX_TO_NAMES,
    hex_to_rgb,
)

EARTH_RADIUS_EQUA = 6378137.0

def convert_rgb_to_names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS2_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'{names[index]}'


def get_entry_point():
    return 'DataAgent'


class DataAgent(AutoPilot):
    """
        Child of the autopilot that additionally runs data collection and storage.
        """
    
    frame_rate = 10.0 
    
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager=None)
        self.weather_tmp = None
        self.weather_name = ""
        self.step_tmp = 0
        self.no_wet = int(os.environ.get('NO_WET', 0))

        self.tm = traffic_manager
        
        self.scenario_name = Path(path_to_conf_file).parent.name
        self.cutin_vehicle_starting_position = None

        self.weathers_ids = list(self.config.weathers)

        if self.save_path is not None and self.datagen:
            (self.save_path / 'lidar').mkdir()
            (self.save_path / 'camera').mkdir()

            (self.save_path / 'camera' / 'rgb_front').mkdir()
            (self.save_path / 'camera' / 'rgb_front_left').mkdir()
            (self.save_path / 'camera' / 'rgb_front_right').mkdir()
            (self.save_path / 'camera' / 'rgb_back').mkdir()
            (self.save_path / 'camera' / 'rgb_back_left').mkdir()
            (self.save_path / 'camera' / 'rgb_back_right').mkdir()
            (self.save_path / 'camera' / 'rgb_top_down').mkdir()

            (self.save_path / 'camera' / 'semantic_front').mkdir()
            (self.save_path / 'camera' / 'semantic_front_left').mkdir()
            (self.save_path / 'camera' / 'semantic_front_right').mkdir()
            (self.save_path / 'camera' / 'semantic_back').mkdir()
            (self.save_path / 'camera' / 'semantic_back_left').mkdir()
            (self.save_path / 'camera' / 'semantic_back_right').mkdir()

            (self.save_path / 'camera' / 'instance_front').mkdir()
            (self.save_path / 'camera' / 'instance_front_left').mkdir()
            (self.save_path / 'camera' / 'instance_front_right').mkdir()
            (self.save_path / 'camera' / 'instance_back').mkdir()
            (self.save_path / 'camera' / 'instance_back_left').mkdir()
            (self.save_path / 'camera' / 'instance_back_right').mkdir()

            (self.save_path / 'camera' / 'depth_front').mkdir()
            (self.save_path / 'camera' / 'depth_front_left').mkdir()
            (self.save_path / 'camera' / 'depth_front_right').mkdir()
            (self.save_path / 'camera' / 'depth_back').mkdir()
            (self.save_path / 'camera' / 'depth_back_left').mkdir()
            (self.save_path / 'camera' / 'depth_back_right').mkdir()

            (self.save_path / 'radar').mkdir()
            (self.save_path / 'anno').mkdir()
            (self.save_path / 'expert_assessment').mkdir()

        self.tmp_visu = int(os.environ.get('TMP_VISU', 0))

        self._active_traffic_light = None
        self.last_lidar = None
        self.last_ego_transform = None

        self.count = 0

    def _init(self, hd_map):
        super()._init(hd_map)
        if self.datagen:
            self.shuffle_weather()

        # obs_config = {
        #         'width_in_pixels': self.config.lidar_resolution_width,
        #         'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0,
        #         'pixels_per_meter': self.config.pixels_per_meter,
        #         'history_idx': [-1],
        #         'scale_bbox': True,
        #         'scale_mask_col': 1.0
        # }

        # self.stop_sign_criteria = RunStopSign(self._world)
        # self.ss_bev_manager = ObsManager(obs_config, self.config)
        # self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=self.stop_sign_criteria)

        # self.ss_bev_manager_augmented = ObsManager(obs_config, self.config)

        # bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
        # transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
        # # Can't clone the carla vehicle object, so I use a dummy class with similar attributes.
        # self.augmented_vehicle_dummy = t_u.CarlaActorDummy(self._vehicle.get_world(), bb_copy, transform_copy,
        #                                                                                                      self._vehicle.id)
        # self.ss_bev_manager_augmented.attach_ego_vehicle(self.augmented_vehicle_dummy,
        #                                                                                                  criteria_stop=self.stop_sign_criteria)
        
        # self._local_planner = LocalPlanner(self._vehicle, opt_dict={}, map_inst=self.world_map)

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        # print("data_agent run_step")
        self.step_tmp += 1

        # Convert LiDAR into the coordinate frame of the ego vehicle
        # input_data['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

        # Must be called before run_step, so that the correct augmentation shift is saved
        # if self.datagen:
        #     self.augment_camera(sensors)

        control = super().run_step(input_data, timestamp, plant=plant)

        tick_data = self.tick(input_data)

        if self.step % self.config.data_save_freq == 0:
            if self.save_path is not None and self.datagen:
                # self.save_sensors(tick_data)
                self.save_b2d_sensors(tick_data)

        # self.last_lidar = input_data['lidar']
        # self.last_ego_transform = self._vehicle.get_transform()
        # these two lines are in tick

        if plant:
            # Control contains data when run with plant
            return {**tick_data, **control}
        else:
            return control

    # def augment_camera(self, sensors):
    #     # Update dummy vehicle
    #     if self.initialized:
    #         # We are still rendering the map for the current frame, so we need to use the translation from the last frame.
    #         last_translation = self.augmentation_translation
    #         last_rotation = self.augmentation_rotation
    #         bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
    #         transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
    #         augmented_loc = transform_copy.transform(carla.Location(0.0, last_translation, 0.0))
    #         transform_copy.location = augmented_loc
    #         transform_copy.rotation.yaw = transform_copy.rotation.yaw + last_rotation
    #         self.augmented_vehicle_dummy.bounding_box = bb_copy
    #         self.augmented_vehicle_dummy.transform = transform_copy

    def shuffle_weather(self):
        # change weather for visual diversity
        if self.weather_tmp is None:
            t = carla.WeatherParameters
            options = dir(t)[:22]
            chosen_preset = random.choice(options)
            self.chosen_preset = chosen_preset
            # print(f"[debug] weather is {str(chosen_preset)}") # debug
            self.weather_name = str(chosen_preset)
            weather = t.__getattribute__(t, chosen_preset)
            if self.no_wet:
                weather.precipitation_deposits = 0
                # weather.wetness = 0 # only affects sensor
            self.weather_tmp = weather

        self._world.set_weather(self.weather_tmp)
        
        # night mode
        vehicles = self._world.get_actors().filter('*vehicle*')
        if self.weather_tmp.sun_altitude_angle < 0.0:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
        else:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState.NONE)

    # def save_sensors(self, tick_data):
    #     frame = self.step // self.config.data_save_freq

    #     # CARLA images are already in opencv's BGR format.
    #     cv2.imwrite(str(self.save_path / 'rgb' / (f'{frame:04}.jpg')), tick_data['rgb'])
    #     cv2.imwrite(str(self.save_path / 'rgb_augmented' / (f'{frame:04}.jpg')), tick_data['rgb_augmented'])

    #     cv2.imwrite(str(self.save_path / 'semantics' / (f'{frame:04}.png')), tick_data['semantics'])
    #     cv2.imwrite(str(self.save_path / 'semantics_augmented' / (f'{frame:04}.png')), tick_data['semantics_augmented'])

    #     cv2.imwrite(str(self.save_path / 'depth' / (f'{frame:04}.png')), tick_data['depth'])
    #     cv2.imwrite(str(self.save_path / 'depth_augmented' / (f'{frame:04}.png')), tick_data['depth_augmented'])

    #     cv2.imwrite(str(self.save_path / 'bev_semantics' / (f'{frame:04}.png')), tick_data['bev_semantics'])
    #     cv2.imwrite(str(self.save_path / 'bev_semantics_augmented' / (f'{frame:04}.png')), tick_data['bev_semantics_augmented'])

    #     # Specialized LiDAR compression format
    #     header = laspy.LasHeader(point_format=self.config.point_format)
    #     header.offsets = np.min(tick_data['lidar'], axis=0)
    #     header.scales = np.array([self.config.point_precision, self.config.point_precision, self.config.point_precision])

    #     with laspy.open(self.save_path / 'lidar' / (f'{frame:04}.laz'), mode='w', header=header) as writer:
    #         point_record = laspy.ScaleAwarePointRecord.zeros(tick_data['lidar'].shape[0], header=header)
    #         point_record.x = tick_data['lidar'][:, 0]
    #         point_record.y = tick_data['lidar'][:, 1]
    #         point_record.z = tick_data['lidar'][:, 2]

    #         writer.write_points(point_record)

    #     with gzip.open(self.save_path / 'boxes' / (f'{frame:04}.json.gz'), 'wt', encoding='utf-8') as f:
    #         json.dump(tick_data['bounding_boxes'], f, indent=4)

    def destroy(self, results=None):
        torch.cuda.empty_cache()

        if results is not None and self.save_path is not None:
            with gzip.open(os.path.join(self.save_path, 'results.json.gz'), 'wt', encoding='utf-8') as f:
                json.dump(results.__dict__, f, indent=2)

        super().destroy(results)
        
        
    def _wps_next_until_lane_end(self, wp):
        try:
            road_id_cur = wp.road_id
            lane_id_cur = wp.lane_id
            road_id_next = road_id_cur
            lane_id_next = lane_id_cur
            curr_wp = [wp]
            next_wps = []
            # https://github.com/carla-simulator/carla/issues/2511#issuecomment-597230746
            while road_id_cur == road_id_next and lane_id_cur == lane_id_next:
                next_wp = curr_wp[0].next(1)
                if len(next_wp) == 0:
                    break
                curr_wp = next_wp
                next_wps.append(next_wp[0])
                road_id_next = next_wp[0].road_id
                lane_id_next = next_wp[0].lane_id
        except:
            next_wps = []
            
        return next_wps

    # def get_bounding_boxes(self, lidar=None):
    #     results = []

    #     ego_transform = self._vehicle.get_transform()
    #     ego_control = self._vehicle.get_control()
    #     ego_velocity = self._vehicle.get_velocity()
    #     ego_matrix = np.array(ego_transform.get_matrix())
    #     ego_rotation = ego_transform.rotation
    #     ego_extent = self._vehicle.bounding_box.extent
    #     ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
    #     ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
    #     ego_yaw = np.deg2rad(ego_rotation.yaw)
    #     ego_brake = ego_control.brake

    #     relative_yaw = 0.0
    #     relative_pos = t_u.get_relative_transform(ego_matrix, ego_matrix)

    #     ego_wp = self.world_map.get_waypoint(self._vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
        
    #     # to compute lane_relative_to_ego for walkers and other cars we first have to precompute some in which direction the opposite lane is & the width of the center lane
    #     left_wp, right_wp = ego_wp.get_left_lane(), ego_wp.get_right_lane()
    #     left_decreasing_lane_id = left_wp is not None and left_wp.lane_id < ego_wp.lane_id or right_wp is not None and right_wp.lane_id > ego_wp.lane_id
        
    #     remove_lanes_for_lane_relative_to_ego = 1
    #     wp = ego_wp
    #     is_opposite = False
    #     while True:
    #         flag = ego_wp.lane_id > 0 and left_decreasing_lane_id or ego_wp.lane_id < 0 and not left_decreasing_lane_id
    #         if is_opposite:
    #             flag = not flag
    #         wp = wp.get_left_lane() if flag else wp.get_right_lane()
                
    #         if wp is None or wp.lane_type == carla.LaneType.Driving and ego_wp.lane_id * wp.lane_id < 0:
    #             break
            
    #         is_opposite = ego_wp.lane_id * wp.lane_id < 0
            
    #         if wp.lane_type != carla.LaneType.Driving:
    #             remove_lanes_for_lane_relative_to_ego += 1

    #     # how far is next junction
    #     next_wps = self._wps_next_until_lane_end(ego_wp)
    #     try:
    #         next_lane_wps_ego = next_wps[-1].next(1)
    #         if len(next_lane_wps_ego) == 0:
    #             next_lane_wps_ego = [next_wps[-1]]
    #     except:
    #         next_lane_wps_ego = []
    #     if ego_wp.is_junction:
    #         distance_to_junction_ego = 0.0
    #         # get distance to ego vehicle
    #     elif len(next_lane_wps_ego)>0 and next_lane_wps_ego[0].is_junction:
    #         distance_to_junction_ego = next_lane_wps_ego[0].transform.location.distance(ego_wp.transform.location)
    #     else:
    #         distance_to_junction_ego = None
            
    #     next_road_ids_ego = []
    #     next_next_road_ids_ego = []
    #     for i, wp in enumerate(next_lane_wps_ego):
    #             next_road_ids_ego.append(wp.road_id)
    #             next_next_wps = self._wps_next_until_lane_end(wp)
    #             try:
    #                 next_next_lane_wps_ego = next_next_wps[-1].next(1)
    #                 if len(next_next_lane_wps_ego) == 0:
    #                     next_next_lane_wps_ego = [next_next_wps[-1]]
    #             except:
    #                 next_next_lane_wps_ego = []
    #             for j, wp2 in enumerate(next_next_lane_wps_ego):
    #                 if wp2.road_id not in next_next_road_ids_ego:
    #                     next_next_road_ids_ego.append(wp2.road_id)

    #     tl = self._world.get_traffic_lights_from_waypoint(ego_wp, 50.0)
    #     if len(tl) == 0:
    #         tl_state = 'None'
    #     else:
    #         tl_state = str(tl[0].state)

    #     ego_lane_direction = ego_wp.lane_id / abs(ego_wp.lane_id)
    #     lanes_to_the_left = []
    #     lanes_to_the_right = []
    #     num_lanes_same_direction = 1 # ego lane
    #     lane_ids_same_direction = [ego_wp.lane_id]
    #     lane_id_left_most_lane_same_direction = ego_wp.lane_id
    #     lane_id_right_most_lane_opposite_direction = ego_wp.lane_id
    #     num_lanes_opposite_direction = 0
    #     shoulder_left = False
    #     shoulder_right = False
    #     parking_left = False
    #     parking_right = False
    #     sidewalk_left = False
    #     sidewalk_right = False
    #     bikelane_left = False
    #     bikelane_right = False

    #     # for loop over left and ride side of the road
    #     for i, lanes in enumerate([lanes_to_the_left, lanes_to_the_right]):
    #         lane_wp = ego_wp
    #         is_road = True
    #         # is_opposite is needed because get_left_lane() returns the left lane from the view point of the lane
    #         # this means if we dont do this and we have a oncoming lane to the left it would just toggle between
    #         # the oncoming lane and the ego lane
    #         is_opposite = False

    #         while is_road:
    #             # first we check for all lanes to the left
    #             if i == 0:
    #                 if not is_opposite:
    #                     lane_wp = lane_wp.get_left_lane()
    #                 else:
    #                     lane_wp = lane_wp.get_right_lane()
    #             # then we check for all lanes to the right
    #             else:
    #                 if not is_opposite:
    #                     lane_wp = lane_wp.get_right_lane()
    #                 else:
    #                     lane_wp = lane_wp.get_left_lane()


    #             if lane_wp is None:
    #                 is_road = False
    #             else:
    #                 direction = lane_wp.lane_id / abs(lane_wp.lane_id)
    #                 lane_type = lane_wp.lane_type
    #                 if lane_type == carla.LaneType.Driving and direction == ego_lane_direction:
    #                     num_lanes_same_direction += 1
    #                     lane_ids_same_direction.append(lane_wp.lane_id)
    #                     if i == 0:
    #                         lane_id_left_most_lane_same_direction = lane_wp.lane_id
    #                 elif lane_type == carla.LaneType.Driving and direction != ego_lane_direction:
    #                     num_lanes_opposite_direction += 1
    #                 elif lane_type == carla.LaneType.Shoulder and i == 0 and lane_wp.lane_width > 1.0:
    #                     shoulder_left = True
    #                 elif lane_type == carla.LaneType.Shoulder and i == 1 and lane_wp.lane_width > 1.0:
    #                     shoulder_right = True
    #                 elif lane_type == carla.LaneType.Parking and i == 0:
    #                     parking_left = True
    #                 elif lane_type == carla.LaneType.Parking and i == 1:
    #                     parking_right = True
    #                 elif lane_type == carla.LaneType.Sidewalk and i == 0:
    #                     sidewalk_left = True
    #                 elif lane_type == carla.LaneType.Sidewalk and i == 1:
    #                     sidewalk_right = True
    #                 elif lane_type == carla.LaneType.Biking and i == 0:
    #                     bikelane_left = True
    #                 elif lane_type == carla.LaneType.Biking and i == 1:
    #                     bikelane_right = True
    #                 else:
    #                     pass


    #                 if direction != ego_lane_direction:
    #                     if is_opposite == False:
    #                         lane_id_right_most_lane_opposite_direction = lane_wp.lane_id
    #                     is_opposite = True
    #                 lanes.append(lane_wp)

    #             # get ego lane number counted from left to right
    #             #https://www.asam.net/standards/detail/opendrive/
    #             # most left should be always the smallest number
    #             min_lane_id = min(lane_ids_same_direction)
    #             ego_lane_number = abs(ego_wp.lane_id - lane_id_left_most_lane_same_direction)

    #     # Check for possible vehicle obstacles
    #     # Retrieve all relevant actors
    #     self._actors = self._world.get_actors()
    #     vehicle_list = self._actors.filter('*vehicle*')

    #     hazard_detected_10 = False
    #     affected_by_vehicle_10, aff_vehicle_id_10, aff_vehicle_dis_10 = self._vehicle_obstacle_detected(vehicle_list, 10)
    #     if affected_by_vehicle_10:
    #             hazard_detected_10 = True
                
    #     hazard_detected_15 = False
    #     affected_by_vehicle_15, aff_vehicle_id_15, aff_vehicle_dis_15 = self._vehicle_obstacle_detected(vehicle_list, 15)
    #     if affected_by_vehicle_15:
    #             hazard_detected_15 = True
                
    #     hazard_detected_20 = False
    #     affected_by_vehicle_20, aff_vehicle_id_20, aff_vehicle_dis_20 = self._vehicle_obstacle_detected(vehicle_list, 20)
    #     if affected_by_vehicle_20:
    #             hazard_detected_20 = True
                
    #     hazard_detected_40 = False
    #     affected_by_vehicle_40, aff_vehicle_id_40, aff_vehicle_dis_40 = self._vehicle_obstacle_detected(vehicle_list, 40)
    #     if affected_by_vehicle_40:
    #             hazard_detected_40 = True
                        
    #     try:
    #         next_action = self.tm.get_next_action(self._vehicle)[0]
    #     except:
    #         next_action = None
    #     result = {
    #             'class': 'ego_car',
    #             'extent': [ego_dx[0], ego_dx[1], ego_dx[2]],
    #             'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #             'yaw': relative_yaw,
    #             'num_points': -1,
    #             'distance': -1,
    #             'speed': ego_speed,
    #             'brake': ego_brake,
    #             'id': int(self._vehicle.id),
    #             'matrix': ego_transform.get_matrix()
    #     }
    #     results.append(result)



    #     for vehicle in vehicle_list:
    #         if vehicle.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
    #             if vehicle.id != self._vehicle.id:
    #                 vehicle_transform = vehicle.get_transform()
    #                 vehicle_rotation = vehicle_transform.rotation
    #                 vehicle_matrix = np.array(vehicle_transform.get_matrix())
    #                 vehicle_control = vehicle.get_control()
    #                 vehicle_velocity = vehicle.get_velocity()
    #                 vehicle_extent = vehicle.bounding_box.extent
    #                 vehicle_id = vehicle.id
    #                 vehicle_wp = self.world_map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
    #                 same_road_as_ego = False
    #                 lane_relative_to_ego = None
    #                 same_direction_as_ego = False
                    
    #                 next_wps = self._wps_next_until_lane_end(vehicle_wp)
    #                 next_lane_wps = next_wps[-1].next(1)
    #                 if len(next_lane_wps) == 0:
    #                     next_lane_wps = [next_wps[-1]]
                    
    #                 next_next_wps = []
    #                 for i, wp in enumerate(next_lane_wps):
    #                     next_next_wps = self._wps_next_until_lane_end(wp)
                    
    #                 try:
    #                     next_next_lane_wps = next_next_wps[-1].next(1)
    #                     if len(next_next_lane_wps) == 0:
    #                         next_next_lane_wps = [next_next_wps[-1]]
    #                 except:
    #                     next_next_lane_wps = []
                    
    #                 if vehicle_wp.is_junction:
    #                     distance_to_junction = 0.0
    #                     # get distance to ego vehicle
    #                 elif next_lane_wps[0].is_junction:
    #                     distance_to_junction = next_lane_wps[0].transform.location.distance(vehicle_wp.transform.location)
    #                 else:
    #                     distance_to_junction = None
                        
    #                 next_road_ids = []
    #                 for i, wp in enumerate(next_lane_wps):
    #                     if wp.road_id not in next_road_ids:
    #                         next_road_ids.append(wp.road_id)
                    
    #                 next_next_road_ids = []
    #                 for i, wp in enumerate(next_next_lane_wps):
    #                     if wp.road_id not in next_next_road_ids:
    #                         next_next_road_ids.append(wp.road_id)
                    
    #                 is_at_traffic_light = vehicle.is_at_traffic_light()

    #                 tl = self._world.get_traffic_lights_from_waypoint(vehicle_wp, 30.0)
    #                 if len(tl) == 0:
    #                     tl_state_vehicle = 'None'
    #                 else:
    #                     tl_state_vehicle = str(tl[0].state)

    #                 if vehicle_wp.road_id == ego_wp.road_id:
    #                     same_road_as_ego = True

    #                     direction = vehicle_wp.lane_id / abs(vehicle_wp.lane_id)
    #                     if direction == ego_lane_direction:
    #                         same_direction_as_ego = True

    #                     lane_relative_to_ego = vehicle_wp.lane_id - ego_wp.lane_id
    #                     lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                        
    #                     if not same_direction_as_ego:
    #                         lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                        
    #                     lane_relative_to_ego = -lane_relative_to_ego

    #                 vehicle_extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
    #                 yaw = np.deg2rad(vehicle_rotation.yaw)

    #                 relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
    #                 relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)
    #                 vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)
    #                 vehicle_brake = vehicle_control.brake
    #                 vehicle_steer = vehicle_control.steer
    #                 vehicle_throttle = vehicle_control.throttle

    #                 # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
    #                 if not lidar is None:
    #                     num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, vehicle_extent_list, lidar)
    #                 else:
    #                     num_in_bbox_points = -1

    #                 distance = np.linalg.norm(relative_pos)
    #                 try:
    #                     rgb = tuple(map(int, vehicle.attributes['color'].split(',')))
    #                     color_name = convert_rgb_to_names(rgb)
    #                 except:
    #                     rgb = None
    #                     color_name = None
    #                 try:
    #                     light_state = vehicle.get_light_state()
    #                 except:
    #                     light_state = 99
    #                 light_state_bin = bin(int(light_state))
    #                 # get positions of 1's
    #                 light_state_bin_pos = [i for i, x in enumerate(reversed(light_state_bin)) if x == '1']
    #                 # get decimal value of 1's
    #                 light_state_dec_pos = [2**i for i in light_state_bin_pos]
    #                 # get VehicleLightState of 1's
    #                 light_state = [carla.VehicleLightState.values[i] for i in light_state_dec_pos]
                    
    #                 try:
    #                     next_action = self.tm.get_next_action(vehicle)[0]
    #                 except:
    #                     next_action = None
                        
                                                
    #                 vehicle_cuts_in = False
    #                 if (self.scenario_name == 'ParkingCutIn') and vehicle.attributes['role_name']=='scenario':
    #                     if self.cutin_vehicle_starting_position is None:
    #                         self.cutin_vehicle_starting_position = vehicle.get_location()

    #                     if vehicle.get_location().distance(self.cutin_vehicle_starting_position) > 0.2 and vehicle.get_location().distance(self.cutin_vehicle_starting_position) < 8: # to make sure the vehicle drives
    #                         vehicle_cuts_in = True
                            
    #                 elif (self.scenario_name == 'StaticCutIn') and vehicle.attributes['role_name']=='scenario':
    #                     if vehicle_speed > 1.0 and abs(vehicle_steer) > 0.2:
    #                         vehicle_cuts_in = True
                    
                            
    #                 result = {
    #                         'class': 'car',
    #                         'color_rgb': rgb,
    #                         'color_name': color_name,
    #                         'next_action': next_action,
    #                         'vehicle_cuts_in': vehicle_cuts_in,
    #                         'road_id': vehicle_wp.road_id,
    #                         'lane_id': vehicle_wp.lane_id,
    #                         'lane_type': vehicle_wp.lane_type,
    #                         'lane_type_str': str(vehicle_wp.lane_type),
    #                         'is_in_junction': vehicle_wp.is_junction,
    #                         'junction_id': vehicle_wp.junction_id,
    #                         'distance_to_junction': distance_to_junction,
    #                         'next_junction_id': next_lane_wps[0].junction_id,
    #                         'next_road_ids': next_road_ids,
    #                         'next_next_road_ids': next_next_road_ids,
    #                         'same_road_as_ego': same_road_as_ego,
    #                         'same_direction_as_ego': same_direction_as_ego,
    #                         'lane_relative_to_ego': lane_relative_to_ego,
    #                         'light_state': light_state_dec_pos,
    #                         'traffic_light_state': tl_state_vehicle,
    #                         'is_at_traffic_light': is_at_traffic_light,
    #                         'base_type': vehicle.attributes['base_type'],
    #                         'role_name': vehicle.attributes['role_name'],
    #                         'number_of_wheels': vehicle.attributes['number_of_wheels'],
    #                         'type_id': vehicle.type_id,
    #                         'extent': vehicle_extent_list,
    #                         'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #                         'yaw': relative_yaw,
    #                         'num_points': int(num_in_bbox_points),
    #                         'distance': distance,
    #                         'speed': vehicle_speed,
    #                         'brake': vehicle_brake,
    #                         'steer': vehicle_steer,
    #                         'throttle': vehicle_throttle,
    #                         'id': int(vehicle_id),
    #                         'matrix': vehicle_transform.get_matrix()
    #                 }
    #                 results.append(result)

    #     walkers = self._actors.filter('*walker*')
    #     for walker in walkers:
    #         if walker.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
    #             walker_transform = walker.get_transform()
    #             walker_velocity = walker.get_velocity()
    #             walker_rotation = walker.get_transform().rotation
    #             walker_matrix = np.array(walker_transform.get_matrix())
    #             walker_id = walker.id
    #             walker_extent = walker.bounding_box.extent
    #             walker_extent = [walker_extent.x, walker_extent.y, walker_extent.z]
    #             yaw = np.deg2rad(walker_rotation.yaw)

    #             relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
    #             relative_pos = t_u.get_relative_transform(ego_matrix, walker_matrix)

    #             walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)

    #             # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
    #             if not lidar is None:
    #                 num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, walker_extent, lidar)
    #             else:
    #                 num_in_bbox_points = -1

    #             distance = np.linalg.norm(relative_pos)
                
    #             walker_wp = self.world_map.get_waypoint(walker.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
    #             lane_type = walker_wp.lane_type
    #             same_road_as_ego = False
    #             lane_relative_to_ego = None
    #             same_direction_as_ego = False
                
    #             if walker_wp.road_id == ego_wp.road_id:
    #                 same_road_as_ego = True

    #                 direction = walker_wp.lane_id / abs(walker_wp.lane_id)
    #                 if direction == ego_lane_direction:
    #                     same_direction_as_ego = True

    #                 lane_relative_to_ego = walker_wp.lane_id - ego_wp.lane_id
    #                 lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                    
    #                 if not same_direction_as_ego:
    #                     lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                    
    #                 lane_relative_to_ego = -lane_relative_to_ego
                

    #             result = {
    #                     'class': 'walker',
    #                     'role_name': walker.attributes['role_name'],
    #                     'gender': walker.attributes['gender'],
    #                     'age': walker.attributes['age'],
    #                     'extent': walker_extent,
    #                     'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #                     'yaw': relative_yaw,
    #                     'num_points': int(num_in_bbox_points),
    #                     'distance': distance,
    #                     'speed': walker_speed,
    #                     'id': int(walker_id),
    #                     'lane_type': lane_type,
    #                     'same_road_as_ego': same_road_as_ego,
    #                     'same_direction_as_ego': same_direction_as_ego,
    #                     'lane_relative_to_ego': lane_relative_to_ego,
    #                     'matrix': walker_transform.get_matrix()
    #             }
    #             results.append(result)

    #     for traffic_light in self.close_traffic_lights:
    #         traffic_light_extent = [traffic_light[0].extent.x, traffic_light[0].extent.y, traffic_light[0].extent.z]

    #         traffic_light_transform = carla.Transform(traffic_light[0].location, traffic_light[0].rotation)
    #         traffic_light_rotation = traffic_light_transform.rotation
    #         traffic_light_matrix = np.array(traffic_light_transform.get_matrix())
    #         yaw = np.deg2rad(traffic_light_rotation.yaw)

    #         relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
    #         relative_pos = t_u.get_relative_transform(ego_matrix, traffic_light_matrix)

    #         distance = np.linalg.norm(relative_pos)

    #         result = {
    #                 'class': 'traffic_light',
    #                 'extent': traffic_light_extent,
    #                 'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #                 'yaw': relative_yaw,
    #                 'distance': distance,
    #                 'state': str(traffic_light[1]),
    #                 'id': int(traffic_light[2]),
    #                 'affects_ego': traffic_light[3],
    #                 'matrix': traffic_light_transform.get_matrix()
    #         }
    #         results.append(result)

    #     for stop_sign in self.close_stop_signs:
    #         stop_sign_extent = [stop_sign[0].extent.x, stop_sign[0].extent.y, stop_sign[0].extent.z]

    #         stop_sign_transform = carla.Transform(stop_sign[0].location, stop_sign[0].rotation)
    #         stop_sign_rotation = stop_sign_transform.rotation
    #         stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
    #         yaw = np.deg2rad(stop_sign_rotation.yaw)

    #         relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
    #         relative_pos = t_u.get_relative_transform(ego_matrix, stop_sign_matrix)

    #         distance = np.linalg.norm(relative_pos)

    #         result = {
    #                 'class': 'stop_sign',
    #                 'extent': stop_sign_extent,
    #                 'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #                 'yaw': relative_yaw,
    #                 'distance': distance,
    #                 'id': int(stop_sign[1]),
    #                 'affects_ego': stop_sign[2],
    #                 'matrix': stop_sign_transform.get_matrix()
    #         }
    #         results.append(result)
            
            
    #     ### stop sign and traffic lights for vqa - add new entry becaus i don't want to mess with tranfuser setting
    #     traffic_lights = self.get_nearby_object(ego_transform.location, self._actors.filter('*light*'), self.config.bb_save_radius)

    #     for traffic_light in traffic_lights:
    #         traffic_light_transform = traffic_light.get_transform()
    #         traffic_light_rotation = traffic_light.get_transform().rotation
    #         traffic_light_matrix = np.array(traffic_light_transform.get_matrix())
    #         traffic_light_id = traffic_light.id
    #         traffic_light_extent = traffic_light.bounding_box.extent
    #         traffic_light_extent = [traffic_light_extent.x, traffic_light_extent.y, traffic_light_extent.z]
    #         yaw = np.deg2rad(traffic_light_rotation.yaw)

    #         relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
    #         relative_pos = t_u.get_relative_transform(ego_matrix, traffic_light_matrix)
            
    #         # continue if light is ehind us
    #         if relative_pos[0] < 0:
    #             continue

    #         # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
    #         if not lidar is None:
    #             num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, traffic_light_extent, lidar)
    #         else:
    #             num_in_bbox_points = -1

    #         distance = np.linalg.norm(relative_pos)
            
    #         same_road_as_ego = False
    #         lane_relative_to_ego = None
    #         same_direction_as_ego = False
    #         trigger = get_trafficlight_trigger_location(traffic_light)
    #         traffic_light_wp = self.world_map.get_waypoint(trigger, project_to_road=False, lane_type=carla.libcarla.LaneType.Any)
            
    #         try:
    #             if traffic_light_wp.road_id == ego_wp.road_id:
    #                 same_road_as_ego = True

    #                 direction = traffic_light_wp.lane_id / abs(traffic_light_wp.lane_id)
    #                 if direction == ego_lane_direction:
    #                     same_direction_as_ego = True

    #                 lane_relative_to_ego = traffic_light_wp.lane_id - ego_wp.lane_id
    #                 lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                    
    #                 if not same_direction_as_ego:
    #                     lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                    
    #                 lane_relative_to_ego = -lane_relative_to_ego
    #         except:
    #             pass
            
    #         try:
    #             road_id = traffic_light_wp.road_id
    #             lane_id = traffic_light_wp.lane_id
    #             junction_id = traffic_light_wp.junction_id
    #         except:
    #             road_id = None
    #             lane_id = None
    #             junction_id = None
    #         result = {
    #                 'class': 'traffic_light_vqa',
    #                 'extent': traffic_light_extent,
    #                 'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #                 'road_id': road_id,
    #                 'lane_id': lane_id,
    #                 'junction_id': junction_id,
    #                 'yaw': relative_yaw,
    #                 'num_points': int(num_in_bbox_points),
    #                 'distance': distance,
    #                 'state_str': str(traffic_light.state),
    #                 'state': int(traffic_light.state),
    #                 'same_road_as_ego': same_road_as_ego,
    #                 'same_direction_as_ego': same_direction_as_ego,
    #                 'affects_ego': same_direction_as_ego,
    #                 'lane_relative_to_ego': lane_relative_to_ego,
    #         }
    #         results.append(result)


    #     stop_signs = self.get_nearby_object(ego_transform.location, self._actors.filter('*stop*'), self.config.bb_save_radius)

    #     for stop_sign in stop_signs:
    #         stop_sign_transform = stop_sign.get_transform()
    #         stop_sign_rotation = stop_sign.get_transform().rotation
    #         stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
    #         stop_sign_id = stop_sign.id
    #         stop_sign_extent = stop_sign.bounding_box.extent
    #         stop_sign_extent = [stop_sign_extent.x, stop_sign_extent.y, stop_sign_extent.z]
    #         yaw = np.deg2rad(stop_sign_rotation.yaw)

    #         relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
    #         relative_pos = t_u.get_relative_transform(ego_matrix, stop_sign_matrix)

    #         # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
    #         if not lidar is None:
    #             num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, stop_sign_extent, lidar)
    #         else:
    #             num_in_bbox_points = -1

    #         distance = np.linalg.norm(relative_pos)
            
    #         same_road_as_ego = False
    #         lane_relative_to_ego = None
    #         same_direction_as_ego = False
    #         trigger = get_trafficlight_trigger_location(stop_sign)
    #         stop_sign_wp = self.world_map.get_waypoint(trigger, project_to_road=False, lane_type=carla.libcarla.LaneType.Any)
            
    #         if stop_sign_wp.road_id == ego_wp.road_id:
    #             same_road_as_ego = True

    #             direction = stop_sign_wp.lane_id / abs(stop_sign_wp.lane_id)
    #             if direction == ego_lane_direction:
    #                 same_direction_as_ego = True

    #             lane_relative_to_ego = stop_sign_wp.lane_id - ego_wp.lane_id
    #             lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                
    #             if not same_direction_as_ego:
    #                 lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                
    #             lane_relative_to_ego = -lane_relative_to_ego
            
    #         result = {
    #                 'class': 'stop_sign_vqa',
    #                 'extent': stop_sign_extent,
    #                 'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #                 'road_id': stop_sign_wp.road_id,
    #                 'lane_id': stop_sign_wp.lane_id,
    #                 'junction_id': stop_sign_wp.junction_id,
    #                 'yaw': relative_yaw,
    #                 'num_points': int(num_in_bbox_points),
    #                 'distance': distance,
    #                 'same_road_as_ego': same_road_as_ego,
    #                 'same_direction_as_ego': same_direction_as_ego,
    #                 'affects_ego': same_direction_as_ego,
    #                 'lane_relative_to_ego': lane_relative_to_ego,
    #         }
    #         results.append(result)


    #     statics = self._actors.filter('static.*')
    #     for static in statics:
    #         if static.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
    #             static_transform = static.get_transform()
    #             static_rotation = static_transform.rotation
    #             static_location = static_transform.location
    #             static_matrix = np.array(static_transform.get_matrix())
    #             static_id = static.id
    #             static_extent = static.bounding_box.extent
    #             #BUG: static_extent x and y are swapped
    #             static_extent = [static_extent.y, static_extent.x, static_extent.z]
    #             yaw = np.deg2rad(static_rotation.yaw)

    #             relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
    #             relative_pos = t_u.get_relative_transform(ego_matrix, static_matrix)
    #             distance = np.linalg.norm(relative_pos)

    #             static_wp = self.world_map.get_waypoint(static_location, project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
    #             same_road_as_ego = False
    #             lane_relative_to_ego = None
    #             same_direction_as_ego = False
    #             if static_wp.road_id == ego_wp.road_id:
    #                 same_road_as_ego = True
                    
    #                 direction = static_wp.lane_id / abs(static_wp.lane_id)
    #                 if direction == ego_lane_direction:
    #                     same_direction_as_ego = True

    #                 lane_relative_to_ego = static_wp.lane_id - ego_wp.lane_id
    #                 lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                    
    #                 if not same_direction_as_ego:
    #                     lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                    
    #                 lane_relative_to_ego = -lane_relative_to_ego


    #             if static.type_id == 'static.prop.mesh':
    #                 if "Car" in static.attributes['mesh_path']:
    #                     result = {
    #                         'class': 'static_car',
    #                         'extent': static_extent,
    #                         'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #                         'yaw': relative_yaw,
    #                         'distance': distance,
    #                         'road_id': static_wp.road_id,
    #                         'junction_id': static_wp.junction_id,
    #                         'lane_id': static_wp.lane_id,
    #                         'on_lane_type': str(static_wp.lane_type),
    #                         'same_road_as_ego': same_road_as_ego,
    #                         'same_direction_as_ego': same_direction_as_ego,
    #                         'lane_relative_to_ego': lane_relative_to_ego,
    #                     }
    #                 else:
    #                     pass
    #             elif static.type_id == 'static.prop.trafficwarning': # the huge traffic warning sign in the scenarios ConstructionObstacle and ConstructionObstacleTwoWays
    #                 result = {
    #                     'class': 'static_trafficwarning',
    #                     'extent': static_extent,
    #                     'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #                     'yaw': relative_yaw,
    #                     'distance': distance,
    #                 }
    #             else:
    #                 result = {
    #                     'class': 'static',
    #                     'type_id': static.type_id,
    #                     'extent': static_extent,
    #                     'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #                     'yaw': relative_yaw,
    #                     'distance': distance,
    #                 }
    #             results.append(result)

    #     landmarks = ego_wp.get_landmarks(40.0)
    #     for landmark in landmarks:
    #         landmark_transform = landmark.transform
    #         landmark_location = landmark_transform.location
    #         landmark_rotation = landmark_transform.rotation
    #         landmark_matrix = np.array(landmark_transform.get_matrix())

    #         yaw = np.deg2rad(landmark_rotation.yaw)

    #         relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
    #         relative_pos = t_u.get_relative_transform(ego_matrix, landmark_matrix)
    #         distance = np.linalg.norm(relative_pos)

    #         result = {
    #             'class': 'landmark',
    #             'name': landmark.name,
    #             'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
    #             'yaw': relative_yaw,
    #             'distance': distance,
    #             'id': int(landmark.id),
    #             'text': landmark.text,
    #             'value': landmark.value,
    #         }
    #         results.append(result)

    #     # weather information:
    #     weather = self._world.get_weather()
    #     weather_info = {
    #         'class': 'weather',
    #         'cloudiness': weather.cloudiness,
    #         'dust_storm': weather.dust_storm,
    #         'fog_density': weather.fog_density,
    #         'fog_distance': weather.fog_distance,
    #         'fog_falloff': weather.fog_falloff,
    #         'mie_scattering_scale': weather.mie_scattering_scale,
    #         'precipitation': weather.precipitation,
    #         'precipitation_deposits': weather.precipitation_deposits,
    #         'rayleigh_scattering_scale': weather.rayleigh_scattering_scale,
    #         'scattering_intensity': weather.scattering_intensity,
    #         'sun_altitude_angle': weather.sun_altitude_angle,
    #         'sun_azimuth_angle': weather.sun_azimuth_angle,
    #         'wetness': weather.wetness,
    #         'wind_intensity': weather.wind_intensity,
    #     }
    #     results.append(weather_info)
        
    #     try:
    #         next_is_junction = next_lane_wps_ego[0].is_junction
    #         next_junction_id = next_lane_wps_ego[0].junction_id
    #     except:
    #         next_is_junction = None
    #         next_junction_id = None

    #     result = {
    #         'class': 'ego_info',
    #         'scenario': self.scenario_name,
    #         'traffic_light_state': tl_state,
    #         'distance_to_junction': distance_to_junction_ego,
    #         'ego_lane_number': ego_lane_number,
    #         'road_id': ego_wp.road_id,
    #         'lane_id': ego_wp.lane_id,
    #         'is_in_junction': ego_wp.is_junction,
    #         'is_intersection': ego_wp.is_intersection,
    #         'junction_id': ego_wp.junction_id,
    #         'next_road_junction': next_is_junction,
    #         'next_junction_id': next_junction_id,
    #         'next_road_ids': next_road_ids_ego,
    #         'next_next_road_ids_ego': next_next_road_ids_ego,
    #         'num_lanes_same_direction': num_lanes_same_direction,
    #         'num_lanes_opposite_direction': num_lanes_opposite_direction,
    #         'lane_change': ego_wp.lane_change,
    #         'lane_change_str': str(ego_wp.lane_change),
    #         'lane_type': ego_wp.lane_type,
    #         'lane_type_str': str(ego_wp.lane_type),
    #         'left_lane_marking_color': ego_wp.left_lane_marking.color,
    #         'left_lane_marking_color_str': str(ego_wp.left_lane_marking.color),
    #         'left_lane_marking_type': ego_wp.left_lane_marking.type,
    #         'left_lane_marking_type_str': str(ego_wp.left_lane_marking.type),
    #         'right_lane_marking_color': ego_wp.right_lane_marking.color,
    #         'right_lane_marking_color_str': str(ego_wp.right_lane_marking.color),
    #         'right_lane_marking_type': ego_wp.right_lane_marking.type,
    #         'right_lane_marking_type_str': str(ego_wp.right_lane_marking.type),
    #         'shoulder_left': shoulder_left,
    #         'shoulder_right': shoulder_right,
    #         'parking_left': parking_left,
    #         'parking_right': parking_right,
    #         'sidewalk_left': sidewalk_left,
    #         'sidewalk_right': sidewalk_right,
    #         'bike_lane_left': bikelane_left,
    #         'bike_lane_right': bikelane_right,
    #         'hazard_detected_10': hazard_detected_10,
    #         'affects_ego_10': aff_vehicle_id_10,
    #         'hazard_detected_15': hazard_detected_15,
    #         'affects_ego_15': aff_vehicle_id_15,
    #         'hazard_detected_20': hazard_detected_20,
    #         'affects_ego_20': aff_vehicle_id_20,
    #         'hazard_detected_40': hazard_detected_40,
    #         'affects_ego_40': aff_vehicle_id_40,
    #     }
    #     results.append(result)


    #     return results

    def get_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, lidar):
        """
        Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
        :param vehicle_pos: Relative position of the vehicle w.r.t. the ego
        :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego
        :param extent: List, Extent of the bounding box
        :param lidar: LiDAR point cloud
        :return: Returns the number of LiDAR hits within the bounding box of the
        vehicle
        """

        rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                                                [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

        # LiDAR in the with the vehicle as origin
        vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

        # check points in bbox
        x, y, z = extent[0], extent[1], extent[2]
        num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                                    (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
        return num_points

    def visualuize(self, rendered, visu_img):
        rendered = cv2.resize(rendered, dsize=(visu_img.shape[1], visu_img.shape[1]), interpolation=cv2.INTER_LINEAR)
        visu_img = cv2.cvtColor(visu_img, cv2.COLOR_BGR2RGB)

        final = np.concatenate((visu_img, rendered), axis=0)

        Image.fromarray(final).save(self.save_path / (f'{self.step:04}.jpg'))


    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        self._use_bbs_detection = False
        self._offset = 0
        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        world_map = self.world_map
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = world_map.get_waypoint(ego_location, lane_type=carla.libcarla.LaneType.Any)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = world_map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle.id, compute_distance(target_vehicle.get_location(), ego_location))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle.id, compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)
    

    def _get_forward_speed(self, transform=None, velocity=None):
        """
        Calculate the forward speed of the vehicle based on its transform and velocity.

        Args:
            transform (carla.Transform, optional): The transform of the vehicle. If not provided, it will be obtained from the vehicle.
            velocity (carla.Vector3D, optional): The velocity of the vehicle. If not provided, it will be obtained from the vehicle.

        Returns:
            float: The forward speed of the vehicle in m/s.
        """
        if not velocity:
            velocity = self._vehicle.get_velocity()

        if not transform:
            transform = self._vehicle.get_transform()

        # Convert the velocity vector to a NumPy array
        velocity_np = np.array([velocity.x, velocity.y, velocity.z])

        # Convert rotation angles from degrees to radians
        pitch_rad = np.deg2rad(transform.rotation.pitch)
        yaw_rad = np.deg2rad(transform.rotation.yaw)

        # Calculate the orientation vector based on pitch and yaw angles
        orientation_vector = np.array([
            np.cos(pitch_rad) * np.cos(yaw_rad), 
            np.cos(pitch_rad) * np.sin(yaw_rad), 
            np.sin(pitch_rad)
        ])

        # Calculate the forward speed by taking the dot product of velocity and orientation vectors
        forward_speed = np.dot(velocity_np, orientation_vector)

        return forward_speed
    
    # below here, from b2d

    def tick(self, input_data):
        # control
        control = self._vehicle.get_control()

        # camera_bgr
        cam_bgr_front = input_data['CAM_FRONT'][1][:, :, :3]
        cam_bgr_front_left = input_data['CAM_FRONT_LEFT'][1][:, :, :3]
        cam_bgr_front_right = input_data['CAM_FRONT_RIGHT'][1][:, :, :3]
        cam_bgr_back = input_data['CAM_BACK'][1][:, :, :3]
        cam_bgr_back_left = input_data['CAM_BACK_LEFT'][1][:, :, :3]
        cam_bgr_back_right = input_data['CAM_BACK_RIGHT'][1][:, :, :3]
        cam_bgr_top_down = input_data['TOP_DOWN'][1][:, :, :3]

        # radar
        radar_front = input_data['RADAR_FRONT'][1].astype(np.float16)
        radar_front_left = input_data['RADAR_FRONT_LEFT'][1].astype(np.float16)
        radar_front_right = input_data['RADAR_FRONT_RIGHT'][1].astype(np.float16)
        radar_back_left = input_data['RADAR_BACK_LEFT'][1].astype(np.float16)
        radar_back_right = input_data['RADAR_BACK_RIGHT'][1].astype(np.float16)

        # lidar
        lidar = input_data['LIDAR_TOP']
        lidar_seg = input_data['LIDAR_TOP_SEG']

        def lidar_to_ego_coordinate(lidar):
            """
            Converts the LiDAR points given by the simulator into the ego agents
            coordinate system
            :param config: GlobalConfig, used to read out lidar orientation and location
            :param lidar: the LiDAR point cloud as provided in the input of run_step
            :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
            coordinate system.
            """
            lidar_rot = [0.0, 0.0, 0.0]
            lidar_pos = [-0.39, 0.0, 1.84]

            yaw = np.deg2rad(lidar_rot[2])
            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

            translation = np.array(lidar_pos)

            # The double transpose is a trick to compute all the points together.
            ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + translation

            return ego_lidar
        
        lidar = lidar_to_ego_coordinate(lidar)
        lidar_360 = lidar
        
        bounding_boxes = self.get_bounding_boxes(lidar=lidar_360, radar=radar_front)
        sensors_anno = self.get_sensors_anno()
        
        self.last_lidar = lidar
        self.last_ego_transform = self._vehicle.get_transform()
        # gps/imu
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        # cam_bgr_depth
        cam_bgr_front_depth = input_data['CAM_FRONT_DEPTH'][1][:, :, :3]
        cam_bgr_front_left_depth = input_data['CAM_FRONT_LEFT_DEPTH'][1][:, :, :3]
        cam_bgr_front_right_depth = input_data['CAM_FRONT_RIGHT_DEPTH'][1][:, :, :3]
        cam_bgr_back_depth = input_data['CAM_BACK_DEPTH'][1][:, :, :3]
        cam_bgr_back_left_depth = input_data['CAM_BACK_LEFT_DEPTH'][1][:, :, :3]
        cam_bgr_back_right_depth = input_data['CAM_BACK_RIGHT_DEPTH'][1][:, :, :3]

        # cam_sem_seg
        cam_front_sem_seg = input_data["CAM_FRONT_SEM_SEG"][1][:, :, 2]
        cam_front_left_sem_seg = input_data["CAM_FRONT_LEFT_SEM_SEG"][1][:, :, 2]
        cam_front_right_sem_seg = input_data["CAM_FRONT_RIGHT_SEM_SEG"][1][:, :, 2]
        cam_back_sem_seg = input_data["CAM_BACK_SEM_SEG"][1][:, :, 2]
        cam_back_left_sem_seg = input_data["CAM_BACK_LEFT_SEM_SEG"][1][:, :, 2]
        cam_back_right_sem_seg = input_data["CAM_BACK_RIGHT_SEM_SEG"][1][:, :, 2]

        # cam_ins_seg
        cam_front_ins_seg = input_data["CAM_FRONT_INS_SEG"][1]
        cam_front_left_ins_seg = input_data["CAM_FRONT_LEFT_INS_SEG"][1]
        cam_front_right_ins_seg = input_data["CAM_FRONT_RIGHT_INS_SEG"][1]
        cam_back_ins_seg = input_data["CAM_BACK_INS_SEG"][1]
        cam_back_left_ins_seg = input_data["CAM_BACK_LEFT_INS_SEG"][1]
        cam_back_right_ins_seg = input_data["CAM_BACK_RIGHT_INS_SEG"][1]
        
        # cam_gray_depth, 16 bit would be ideal, but we can't afford the extra storage.
        cam_gray_front_depth = convert_depth(cam_bgr_front_depth)
        cam_gray_front_left_depth = convert_depth(cam_bgr_front_left_depth)
        cam_gray_front_right_depth = convert_depth(cam_bgr_front_right_depth)
        cam_gray_back_depth = convert_depth(cam_bgr_back_depth)
        cam_gray_back_left_depth = convert_depth(cam_bgr_back_left_depth)
        cam_gray_back_right_depth = convert_depth(cam_bgr_back_right_depth)
        
        # weather
        weather = self.weather_name

        self.cam_bgr_mapping = {
            'CAM_FRONT': 'cam_bgr_front',
            'CAM_FRONT_LEFT': 'cam_bgr_front_left',
            'CAM_FRONT_RIGHT': 'cam_bgr_front_right',
            'CAM_BACK': 'cam_bgr_back',
            'CAM_BACK_LEFT': 'cam_bgr_back_left',
            'CAM_BACK_RIGHT': 'cam_bgr_back_right',
        }

        self.cam_bgr_depth_mapping = {
            'CAM_FRONT': 'cam_bgr_front_depth',
            'CAM_FRONT_LEFT': 'cam_bgr_front_left_depth',
            'CAM_FRONT_RIGHT': 'cam_bgr_front_right_depth',
            'CAM_BACK': 'cam_bgr_back_depth',
            'CAM_BACK_LEFT': 'cam_bgr_back_left_depth',
            'CAM_BACK_RIGHT': 'cam_bgr_back_right_depth',

        }

        self.cam_gray_depth_mapping = {
            'CAM_FRONT': 'cam_gray_front_depth',
            'CAM_FRONT_LEFT': 'cam_gray_front_left_depth',
            'CAM_FRONT_RIGHT': 'cam_gray_front_right_depth',
            'CAM_BACK': 'cam_gray_back_depth',
            'CAM_BACK_LEFT': 'cam_gray_back_left_depth',
            'CAM_BACK_RIGHT': 'cam_gray_back_right_depth',
        }

        self.cam_seg_mapping = {
            'CAM_FRONT': 'cam_front_sem_seg',
            'CAM_FRONT_LEFT': 'cam_front_left_sem_seg',
            'CAM_FRONT_RIGHT': 'cam_front_right_sem_seg',
            'CAM_BACK': 'cam_back_sem_seg',
            'CAM_BACK_LEFT': 'cam_back_left_sem_seg',
            'CAM_BACK_RIGHT': 'cam_back_right_sem_seg',
        }

        self.cam_ins_mapping = {
            'CAM_FRONT': 'cam_front_ins_seg',
            'CAM_FRONT_LEFT': 'cam_front_left_ins_seg',
            'CAM_FRONT_RIGHT': 'cam_front_right_ins_seg',
            'CAM_BACK': 'cam_back_ins_seg',
            'CAM_BACK_LEFT': 'cam_back_left_ins_seg',
            'CAM_BACK_RIGHT': 'cam_back_right_ins_seg',
        }

        self.radar_mapping = {
            'RADAR_FRONT': 'radar_front',
            'RADAR_FRONT_LEFT': 'radar_front_left',
            'RADAR_FRONT_RIGHT': 'radar_front_right',
            'RADAR_BACK_LEFT': 'radar_back_left',
            'RADAR_BACK_RIGHT': 'radar_back_right',
        }

        self.cam_yaw_mapping = {
            'CAM_FRONT': 0.0,
            'CAM_FRONT_LEFT': -55.0,
            'CAM_FRONT_RIGHT': 55.0,
            'CAM_BACK': 180.0,
            'CAM_BACK_LEFT': -110.0,
            'CAM_BACK_RIGHT': 110.0,
        }

        results = {
                # cam_bgr
                'cam_bgr_front': cam_bgr_front,
                'cam_bgr_front_left': cam_bgr_front_left,
                'cam_bgr_front_right': cam_bgr_front_right,
                'cam_bgr_back': cam_bgr_back,
                'cam_bgr_back_left': cam_bgr_back_left,
                'cam_bgr_back_right': cam_bgr_back_right,
                'cam_bgr_top_down': cam_bgr_top_down,
                # cam_sem_seg
                'cam_front_sem_seg': cam_front_sem_seg,
                'cam_front_left_sem_seg': cam_front_left_sem_seg,
                'cam_front_right_sem_seg': cam_front_right_sem_seg,
                'cam_back_sem_seg': cam_back_sem_seg,
                'cam_back_left_sem_seg': cam_back_left_sem_seg,
                'cam_back_right_sem_seg': cam_back_right_sem_seg,
                # cam_ins_seg
                'cam_front_ins_seg': cam_front_ins_seg,
                'cam_front_left_ins_seg': cam_front_left_ins_seg,
                'cam_front_right_ins_seg': cam_front_right_ins_seg,
                'cam_back_ins_seg': cam_back_ins_seg,
                'cam_back_left_ins_seg': cam_back_left_ins_seg,
                'cam_back_right_ins_seg': cam_back_right_ins_seg,

                # cam_gray_depth
                # save the original bgr depth, please remember to post-process the depth
                'cam_bgr_front_depth': cam_bgr_front_depth,
                'cam_bgr_front_left_depth' : cam_bgr_front_left_depth,
                'cam_bgr_front_right_depth': cam_bgr_front_right_depth,
                'cam_bgr_back_depth': cam_bgr_back_depth,
                'cam_bgr_back_left_depth': cam_bgr_back_left_depth,
                'cam_bgr_back_right_depth': cam_bgr_back_right_depth,
                
                'cam_gray_front_depth': cam_gray_front_depth,
                'cam_gray_front_left_depth': cam_gray_front_left_depth,
                'cam_gray_front_right_depth': cam_gray_front_right_depth,
                'cam_gray_back_depth': cam_gray_back_depth,
                'cam_gray_back_left_depth': cam_gray_back_left_depth,
                'cam_gray_back_right_depth': cam_gray_back_right_depth,
                
                # radar
                'radar_front': radar_front,
                'radar_front_left': radar_front_left,
                'radar_front_right': radar_front_right,
                'radar_back_left': radar_back_left,
                'radar_back_right': radar_back_right,
                # lidar
                'lidar' : lidar_360,
                'lidar_seg': lidar_seg,
                # other
                'pos': self._vehicle.get_location(),
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'weather': weather,
                "acceleration":acceleration,
                "angular_velocity":angular_velocity,
                'bounding_boxes': bounding_boxes,
                'sensors_anno': sensors_anno,
                'throttle': control.throttle,
                'steer': control.steer,
                'brake': control.brake,
                'reverse': control.reverse,
                'town': self.world_map.name,
                }
        return results
    
    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]
        id_ = sensor_spec["id"]
        attributes = {}

        if type_ == 'sensor.opendrive_map':
            attributes['reading_frequency'] = sensor_spec['reading_frequency']
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.speedometer':
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            attributes['reading_frequency'] = 1 / delta_time
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_ == 'sensor.camera.rgb':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])
            attributes['role_name'] = str(sensor_spec['id'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        
        elif type_ == 'sensor.camera.depth':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])
            attributes['role_name'] = str(sensor_spec['id'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                roll=sensor_spec['roll'],
                                                yaw=sensor_spec['yaw'])
        
        elif type_ == 'sensor.camera.semantic_segmentation':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])
            attributes['role_name'] = str(sensor_spec['id'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                roll=sensor_spec['roll'],
                                                yaw=sensor_spec['yaw'])
        
        elif type_ == 'sensor.camera.instance_segmentation':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])
            attributes['role_name'] = str(sensor_spec['id'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                roll=sensor_spec['roll'],
                                                yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast':
            attributes['range'] = str(sensor_spec['range'])
            attributes['rotation_frequency'] = str(sensor_spec['rotation_frequency'])
            attributes['channels'] = str(sensor_spec['channels'])
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(sensor_spec['points_per_second'])
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(sensor_spec['dropoff_general_rate'])
            attributes['dropoff_intensity_limit'] = str(sensor_spec['dropoff_intensity_limit'])
            attributes['dropoff_zero_intensity'] = str(sensor_spec['dropoff_zero_intensity'])
            attributes['role_name'] = str(sensor_spec['id'])


            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast_semantic':
            attributes['range'] = str(sensor_spec['range'])
            attributes['rotation_frequency'] = str(sensor_spec['rotation_frequency'])
            attributes['channels'] = str(sensor_spec['channels'])
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(sensor_spec['points_per_second'])
            attributes['role_name'] = str(sensor_spec['id'])


            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.radar':
            attributes['horizontal_fov'] = str(sensor_spec['horizontal_fov'])  # degrees
            attributes['vertical_fov'] = str(sensor_spec['vertical_fov'])  # degrees
            attributes['points_per_second'] = '1500'
            attributes['range'] = sensor_spec['range']  # meters
            attributes['role_name'] = str(sensor_spec['id'])


            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.gnss':
            attributes['noise_alt_stddev'] = str(0.000005)
            attributes['noise_lat_stddev'] = str(0.000005)
            attributes['noise_lon_stddev'] = str(0.000005)
            attributes['noise_alt_bias'] = str(0.0)
            attributes['noise_lat_bias'] = str(0.0)
            attributes['noise_lon_bias'] = str(0.0)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation()
            attributes['role_name'] = str(sensor_spec['id'])


        elif type_ == 'sensor.other.imu':
            attributes['noise_accel_stddev_x'] = str(0.001)
            attributes['noise_accel_stddev_y'] = str(0.001)
            attributes['noise_accel_stddev_z'] = str(0.015)
            attributes['noise_gyro_stddev_x'] = str(0.001)
            attributes['noise_gyro_stddev_y'] = str(0.001)
            attributes['noise_gyro_stddev_z'] = str(0.001)
            attributes['role_name'] = str(sensor_spec['id'])

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes
    
    def setup_sensors(self):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        vehicle = self._vehicle
        self.sensor_interface = SensorInterface()
        world = CarlaDataProvider.get_world()
        bp_library = world.get_blueprint_library()
        for sensor_spec in self.sensors():
            type_, id_, sensor_transform, attributes = self._preprocess_sensor_spec(sensor_spec)

            # These are the pseudosensors (not spawned)
            if type_ == 'sensor.opendrive_map':
                sensor = OpenDriveMapReader(vehicle, attributes['reading_frequency'])
            elif type_ == 'sensor.speedometer':
                sensor = SpeedometerReader(vehicle, attributes['reading_frequency'])

            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(type_)
                for key, value in attributes.items():
                    bp.set_attribute(str(key), str(value))
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)

            # setup callback
            sensor.listen(CallBack(id_, type_, sensor, self.sensor_interface))
            self._sensors_list.append(sensor)

        # Some sensors miss sending data during the first ticks, so tick several times and remove the data
        for _ in range(10):
            world.tick()
    
    def sensors(self):
        sensors = [
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                    },
                # camera depth 
                {
                    'type': 'sensor.camera.depth',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT_DEPTH'
                    },
                # camera seg
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT_SEM_SEG'
                    },
                # camera seg
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT_INS_SEG'
                    },
                # lidar
                {   'type': 'sensor.lidar.ray_cast',
                    'x': -0.39, 'y': 0.0, 'z': 1.84,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'range': 85,
                    'rotation_frequency': 10,
                    'channels': 64,
                    'points_per_second': 600000,
                    'dropoff_general_rate': 0.0,
                    'dropoff_intensity_limit': 0.0,
                    'dropoff_zero_intensity': 0.0,
                    'id': 'LIDAR_TOP'
                    },
                {   'type': 'sensor.lidar.ray_cast_semantic',
                    'x': -0.39, 'y': 0.0, 'z': 1.84,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'range': 85,
                    'rotation_frequency': 10,
                    'channels': 64,
                    'points_per_second': 600000,
                    'id': 'LIDAR_TOP_SEG'
                    },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                    },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                    },
                # radar
                {
                    'type': 'sensor.other.radar', 
                    'x': 2.27, 'y': 0.0, 'z': 0.48, 
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                    'id': 'RADAR_FRONT'
                    },
                {
                    'type': 'sensor.other.radar', 
                    'x': 1.21, 'y': -0.85, 'z': 0.74, 
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                    'id': 'RADAR_FRONT_LEFT'
                    },
                {
                    'type': 'sensor.other.radar', 
                    'x': 1.21, 'y': 0.85, 'z': 0.74, 
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
                    'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                    'id': 'RADAR_FRONT_RIGHT'
                    },
                {
                    'type': 'sensor.other.radar', 
                    'x': -2.0, 'y': -0.67, 'z': 0.51, 
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                    'id': 'RADAR_BACK_LEFT'
                    },
                {
                    'type': 'sensor.other.radar', 
                    'x': -2.0, 'y': 0.67, 'z': 0.51, 
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                    'id': 'RADAR_BACK_RIGHT'
                    },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                    },

                ### Debug sensor, not used by the model
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'TOP_DOWN'
                    },
            ]
        self.sensors_mapping = {}
        for sensor in sensors:
            self.sensors_mapping[sensor['id']] = sensor
        return sensors

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale
        return gps
    
    def get_target_gps(self, gps, compass):
		# target gps
        def gps_to_location(gps):
            # gps content: numpy array: [lat, lon, alt]
            lat, lon, z = gps
            scale = math.cos(self.lat_ref * math.pi / 180.0)
            my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
            mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
            y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
            x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
            z = float(z)
            location = carla.Location(x=x, y=y, z=z)
            return location
            pass
        global_plan_gps = self._global_plan[:]
        next_gps, _ = global_plan_gps[min(self.navigation_idx+1, len(global_plan_gps)-1)]
        next_gps = np.array([next_gps['lat'], next_gps['lon'], next_gps['z']])
        next_vec_in_global = gps_to_location(next_gps) - gps_to_location(gps)
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
        loc_in_ev = vec_global_to_ref(next_vec_in_global, ref_rot_in_global)
        
        if np.sqrt(loc_in_ev.x**2+loc_in_ev.y**2) < 12.0 and loc_in_ev.x < 0.0:
            self.navigation_idx += 1
        
        self.navigation_idx = min(self.navigation_idx, len(self._global_plan)-2)
        _, road_option_0 = global_plan_gps[max(0, self.navigation_idx)]
        gps_point, road_option_1 = global_plan_gps[self.navigation_idx+1]
        gps_point = np.array([gps_point['lat'], gps_point['lon'], gps_point['z']])
        if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option = road_option_1
        else:
            road_option = road_option_0

        return np.array(gps_point, dtype=np.float32), np.array([road_option.value], dtype=np.int8)

    def save_b2d_sensors(self, tick_data):
        # print("[debug] save_b2d_sensor called") # debug
        frame = self.count
        # CARLA images are already in opencv's BGR format.
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_front' / (f'{frame:05}.jpg')), tick_data['cam_bgr_front'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_front_left' / (f'{frame:05}.jpg')), tick_data['cam_bgr_front_left'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_front_right' / (f'{frame:05}.jpg')), tick_data['cam_bgr_front_right'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_back' / (f'{frame:05}.jpg')), tick_data['cam_bgr_back'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_back_left' / (f'{frame:05}.jpg')), tick_data['cam_bgr_back_left'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_back_right' / (f'{frame:05}.jpg')), tick_data['cam_bgr_back_right'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_top_down' / (f'{frame:05}.jpg')), tick_data['cam_bgr_top_down'], [cv2.IMWRITE_JPEG_QUALITY, 20])

        cv2.imwrite(str(self.save_path / 'camera' / 'semantic_front' / (f'{frame:05}.png')), tick_data['cam_front_sem_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'semantic_front_left' / (f'{frame:05}.png')), tick_data['cam_front_left_sem_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'semantic_front_right' / (f'{frame:05}.png')), tick_data['cam_front_right_sem_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'semantic_back' / (f'{frame:05}.png')), tick_data['cam_back_sem_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'semantic_back_left' / (f'{frame:05}.png')), tick_data['cam_back_left_sem_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'semantic_back_right' / (f'{frame:05}.png')), tick_data['cam_back_right_sem_seg'])

        cv2.imwrite(str(self.save_path / 'camera' / 'instance_front' / (f'{frame:05}.png')), tick_data['cam_front_ins_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'instance_front_left' / (f'{frame:05}.png')), tick_data['cam_front_left_ins_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'instance_front_right' / (f'{frame:05}.png')), tick_data['cam_front_right_ins_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'instance_back' / (f'{frame:05}.png')), tick_data['cam_back_ins_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'instance_back_left' / (f'{frame:05}.png')), tick_data['cam_back_left_ins_seg'])
        cv2.imwrite(str(self.save_path / 'camera' / 'instance_back_right' / (f'{frame:05}.png')), tick_data['cam_back_right_ins_seg'])

        cv2.imwrite(str(self.save_path / 'camera' / 'depth_front' / (f'{frame:05}.png')), tick_data['cam_gray_front_depth'])
        cv2.imwrite(str(self.save_path / 'camera' / 'depth_front_left' / (f'{frame:05}.png')), tick_data['cam_gray_front_left_depth'])
        cv2.imwrite(str(self.save_path / 'camera' / 'depth_front_right' / (f'{frame:05}.png')), tick_data['cam_gray_front_right_depth'])
        cv2.imwrite(str(self.save_path / 'camera' / 'depth_back' / (f'{frame:05}.png')), tick_data['cam_gray_back_depth'])
        cv2.imwrite(str(self.save_path / 'camera' / 'depth_back_left' / (f'{frame:05}.png')), tick_data['cam_gray_back_left_depth'])
        cv2.imwrite(str(self.save_path / 'camera' / 'depth_back_right' / (f'{frame:05}.png')), tick_data['cam_gray_back_right_depth'])
        
        with h5py.File(str(self.save_path / 'radar' / (f'{frame:05}.h5')), 'w') as f:
            f.create_dataset('radar_front', data=tick_data['radar_front'], compression='gzip', compression_opts=9, chunks=True)
            f.create_dataset('radar_front_left', data=tick_data['radar_front_left'], compression='gzip', compression_opts=9, chunks=True)
            f.create_dataset('radar_front_right', data=tick_data['radar_front_right'], compression='gzip', compression_opts=9, chunks=True)
            f.create_dataset('radar_back_left', data=tick_data['radar_back_left'], compression='gzip', compression_opts=9, chunks=True)
            f.create_dataset('radar_back_right', data=tick_data['radar_back_right'], compression='gzip', compression_opts=9, chunks=True)

        # Specialized LiDAR compression format
        header = laspy.LasHeader(point_format=0)  # LARS point format used for storing
        header.offsets = np.min(tick_data['lidar'], axis=0)
        point_precision = 0.001
        header.scales = np.array([point_precision, point_precision, point_precision])

        with laspy.open(self.save_path / 'lidar' / (f'{frame:05}.laz'), mode='w', header=header) as writer:
            point_record = laspy.ScaleAwarePointRecord.zeros(tick_data['lidar'].shape[0], header=header)
            point_record.x = tick_data['lidar'][:, 0]
            point_record.y = tick_data['lidar'][:, 1]
            point_record.z = tick_data['lidar'][:, 2]
            writer.write_points(point_record)

        anno_data = {
                'x': tick_data['pos'].x,
                'y': tick_data['pos'].y,
                'throttle': tick_data['throttle'],
                'steer': tick_data['steer'],
                'brake': tick_data['brake'],
                'reverse': tick_data['reverse'],
                'theta': tick_data['compass'],
                'speed': tick_data['speed'],
                # 'x_command_far': far_node[0],
                # 'y_command_far': far_node[1],
                # 'command_far': far_command.value,
                # 'x_command_near': near_node[0],
                # 'y_command_near': near_node[1],
                # 'command_near': near_command.value,
                # 'should_brake': should_brake,
                # 'x_target': tick_data['x_target'],
                # 'y_target': tick_data['y_target'],
                # 'target_command': tick_data['target_command'].tolist(),
                # 'target_gps': tick_data['target_gps'].tolist(),
                # 'next_command': tick_data['next_command'],
                'weather': tick_data['weather'],
                "acceleration":tick_data["acceleration"].tolist(),
                "angular_velocity":tick_data["angular_velocity"].tolist(),
                'bounding_boxes': tick_data['bounding_boxes'],
                'sensors': tick_data['sensors_anno'],
                # 'only_ap_brake': tick_data['only_ap_brake'],
                }
        with gzip.open(self.save_path / 'anno' / f'{frame:05}.json.gz', 'wt', encoding='utf-8') as f:
            json.dump(anno_data, f, indent=4)
        # if self.count > -1:
        #     np.savez(self.save_path / 'expert_assessment' / f'{frame-1:05}.npz', np.concatenate((self.feature, self.value, np.array([self.action_index], dtype=np.float32))))
        self.count += 1

    # add feature (yzj)
    def _point_inside_boundingbox(self, point, bb_center, bb_extent, multiplier=1.5):
        A = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
        B = carla.Vector2D(bb_center.x + multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
        D = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y + multiplier * bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad 
    
    def affected_by_traffic_light(self, traffic_light, center, window_size=50):
        inx = min(window_size, len(self.route_waypoints))
        # print(self.route_waypoints)
        if inx <= 1:
            return False
        for wp in self.route_waypoints[:inx]:
            if self._point_inside_boundingbox(wp.transform.location, center, traffic_light.trigger_volume.extent):
                return True
        return False

    def get_traffic_color(self, state):
        if state == carla.libcarla.TrafficLightState.Green:
            return 'green'
        if state == carla.libcarla.TrafficLightState.Yellow:
            return 'yellow'
        if state == carla.libcarla.TrafficLightState.Red:
            return 'red'
        if state == carla.libcarla.TrafficLightState.Unknown:
            return 'unknown'
        if state == carla.libcarla.TrafficLightState.Off:
            return 'off'
        raise Exception(f"{state} not in Green, Yellow, Red, Unknown, Off")
    
    def get_affect_sign(self, actors):
        all_actors = []
        affect_signs = []
        mini_sign = DIS_SIGN_SAVE + 1
        most_affect_sign = None
        # find all lights
        ego_vehicle_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
        for sign in actors:
            flag = 0
            sign_loc = sign.get_location()
            if compute_2d_distance(sign_loc, self._vehicle.get_transform().location) > DIS_SIGN_SAVE:
                continue
            all_actors.append(sign)

            # find all affect lights 
            if hasattr(sign, 'trigger_volume'):
                sign_vol_loc = sign.trigger_volume.location
                sign_vol_loc_world = sign.get_transform().transform(sign_vol_loc)
                sign_vol_loc_world_wp = self.world_map.get_waypoint(sign_vol_loc_world)
                while not sign_vol_loc_world_wp.is_intersection:
                    if len(sign_vol_loc_world_wp.next(0.5)) > 0:
                        next_sign_vol_loc_world_wp = sign_vol_loc_world_wp.next(0.5)[0]
                    else:
                        flag = 1
                        break
                    if next_sign_vol_loc_world_wp and not next_sign_vol_loc_world_wp.is_intersection:
                        sign_vol_loc_world_wp = next_sign_vol_loc_world_wp
                    else:
                        break
                if flag:
                    continue
                if self.affected_by_traffic_light(sign, carla.Location(x=sign_vol_loc_world_wp.transform.location.x, y=sign_vol_loc_world_wp.transform.location.y, z=0)):
                    affect_signs.append(sign)
                    dis = np.abs(compute_2d_distance(ego_vehicle_waypoint.transform.location, sign.get_transform().transform(sign.trigger_volume.location)))
                    if dis < mini_sign:
                        most_affect_sign = sign
                        mini_sign = dis
            else:
                sign_vol_loc = sign.get_transform().location
                sign_vol_loc_world_wp = self.world_map.get_waypoint(sign_vol_loc)
                dis = compute_2d_distance(sign_vol_loc_world_wp.transform.location, ego_vehicle_waypoint.transform.location)
                forward_vec = self._vehicle.get_transform().get_forward_vector()
                ray = sign_vol_loc_world_wp.transform.location - self._vehicle.get_location()
                if forward_vec.dot(ray) < 0:
                    continue
                if dis < mini_sign:
                    most_affect_sign = sign
                    mini_sign = dis
        return all_actors, most_affect_sign

    def get_actor_filter_traffic_sign(self):
        actor_data = EasyDict({})
        speed_limit_sign = list(CarlaDataProvider.get_world().get_actors().filter("*traffic.speed_limit*")) # carla.libcarla.TrafficSign
        stop_sign = list(CarlaDataProvider.get_world().get_actors().filter("*traffic.stop*")) # carla.libcarla.TrafficSign
        yield_sign = list(CarlaDataProvider.get_world().get_actors().filter("*traffic.yield*")) # carla.libcarla.TrafficSign
        warning = list(CarlaDataProvider.get_world().get_actors().filter('*warning*'))
        dirtdebris = list(CarlaDataProvider.get_world().get_actors().filter('*dirtdebris*'))
        cone = list(CarlaDataProvider.get_world().get_actors().filter('*cone*'))

        actors = speed_limit_sign + stop_sign + yield_sign + warning + dirtdebris + cone
        all_actors, most_affect_sign = self.get_affect_sign(actors)
        actor_data.actors = all_actors
        actor_data.most_affect_sign = most_affect_sign
        return actor_data

    def get_actor_filter_traffic_light(self):
        actor_data = EasyDict({})
        lights = CarlaDataProvider.get_world().get_actors().filter("*traffic_light*")
        all_lights = []
        affect_lights = []
        most_affect_light = None
        mini_lt = DIS_LIGHT_SAVE + 1

        # find all lights
        for lt in lights:
            flag = 0
            lt_loc = lt.get_location()
            if compute_2d_distance(lt_loc, self._vehicle.get_location()) > DIS_LIGHT_SAVE: # lidar range
                continue
            all_lights.append(lt)

            # find all affect lights 
            lt_vol_loc = lt.trigger_volume.location
            lt_vol_loc_world = lt.get_transform().transform(lt_vol_loc)
            lt_vol_loc_world_wp = self.world_map.get_waypoint(lt_vol_loc_world)
            while not lt_vol_loc_world_wp.is_intersection:
                if len(lt_vol_loc_world_wp.next(0.5)) > 0:
                    next_lt_vol_loc_world_wp = lt_vol_loc_world_wp.next(0.5)[0]
                else:
                    flag = 1
                    break 
                if next_lt_vol_loc_world_wp and not next_lt_vol_loc_world_wp.is_intersection:
                    lt_vol_loc_world_wp = next_lt_vol_loc_world_wp
                else:
                    break
            if flag:
                continue
            if self.affected_by_traffic_light(lt, carla.Location(x=lt_vol_loc_world_wp.transform.location.x, y=lt_vol_loc_world_wp.transform.location.y, z=0)):
                affect_lights.append(lt)
                # find most affect light_actor, min_dis=DIS_LIGHT_SAVE
                dis = np.abs(compute_2d_distance(lt.get_transform().transform(lt.trigger_volume.location), self._vehicle.get_location()))
                forward_vec = self._vehicle.get_transform().get_forward_vector()
                ray = lt.get_transform().transform(lt.trigger_volume.location) - self._vehicle.get_location()
                if forward_vec.dot(ray) < 0:
                    continue
                if dis < mini_lt:
                    most_affect_light = lt
                    mini_lt = dis
        
        actor_data.lights = all_lights
        actor_data.affect_lights = affect_lights
        actor_data.most_affect_light = most_affect_light

        #  get distance
        if most_affect_light is not None:
            trigger_volume = most_affect_light.trigger_volume
            box = trigger_volume.extent
            loc = trigger_volume.location
            ori = trigger_volume.rotation.get_forward_vector()
            trigger_loc = [loc.x, loc.y, loc.z]
            trigger_ori = [ori.x, ori.y, ori.z]
            trigger_box = [box.x, box.y]

            world_loc = most_affect_light.get_transform().transform(loc)
            world_loc_wp = self.world_map.get_waypoint(world_loc)
            while not world_loc_wp.is_intersection:
                next_world_loc_wp = world_loc_wp.next(0.5)[0]
                if next_world_loc_wp and not next_world_loc_wp.is_intersection:
                    world_loc_wp = next_world_loc_wp
                else:
                    break
            
            world_loc_wp = carla.Location(x=world_loc_wp.transform.location.x, y=world_loc_wp.transform.location.y, z=0)
            pos = self._vehicle.get_location()
            pos = carla.Location(x=pos.x, y=pos.y, z=0)

            # ego2lane_dis = world_loc_wp.distance(pos)
            ego2lane_dis = compute_2d_distance(world_loc_wp, pos)
            ego2light_dis = compute_2d_distance(most_affect_light.get_location(), self._vehicle.get_location())
            most_affect_light_id = most_affect_light.id
            most_affect_light_state = self.get_traffic_color(most_affect_light.state)

            # record data
            actor_data.most_affect_light = EasyDict()
            actor_data.most_affect_light.id = most_affect_light_id
            actor_data.most_affect_light.state = most_affect_light_state
            actor_data.most_affect_light.ego2lane_dis = ego2lane_dis
            actor_data.most_affect_light.ego2light_dis = ego2light_dis
            actor_data.most_affect_light.trigger_volume = EasyDict()
            actor_data.most_affect_light.trigger_volume.location = trigger_loc
            actor_data.most_affect_light.trigger_volume.orientation = trigger_ori
            actor_data.most_affect_light.trigger_volume.extent = trigger_box
        return actor_data 

        
    def get_actor_filter_vehicle(self):
        vehicles_dict = EasyDict({})

        world = CarlaDataProvider.get_world()
        vehicles = world.get_actors().filter('*vehicle*')
        vehicles_list = []
        for actor in vehicles:
            dist = compute_2d_distance(actor.get_transform().location, self._vehicle.get_transform().location)
            # Filter for the vehicles within DIS_CAR_SAVE m
            if dist < DIS_CAR_SAVE: # lidar range
                vehicles_list.append(actor)
        vehicles_dict.vehicle = vehicles_list

        others = world.get_actors().filter('*static.prop.mesh*')
        static_list = []
        for actor in others:
            # filter static vehicle
            mesh_path = actor.attributes['mesh_path'].split('/Game/Carla/')[1]
            if 'Car' in mesh_path or 'Truck' in mesh_path or 'Bus' in mesh_path or 'Motorcycle' in mesh_path or 'Bicycle' in mesh_path:
                dist = compute_2d_distance(actor.get_transform().location, self._vehicle.get_transform().location)
                # Filter for the vehicles within DIS_CAR_SAVE
                if dist < DIS_CAR_SAVE: # lidar range
                    static_list.append(actor)
        vehicles_dict.static = static_list
        return vehicles_dict
    

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed
    
    def get_sensors_anno(self):
        results = {}
        sensors = {}
        world = CarlaDataProvider.get_world()

        for value in world.get_actors().filter('*sensor.camera.rgb'):
            sensors[value.attributes['role_name']] = value

        for key in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'TOP_DOWN']:
            value = sensors[key]
            location = value.get_transform().location
            rotation = value.get_transform().rotation
            world2cam = value.get_transform().get_inverse_matrix()
            width = int(value.attributes['image_size_x'])
            height = int(value.attributes['image_size_y'])
            fov = float(value.attributes['fov'])
            K = build_projection_matrix(width, height, fov=fov)
            cam2ego = get_matrix(location=[self.sensors_mapping[key]['x'], self.sensors_mapping[key]['y'], self.sensors_mapping[key]['z']], 
                                 rotation=[self.sensors_mapping[key]['pitch'], self.sensors_mapping[key]['roll'], self.sensors_mapping[key]['yaw']])
            result ={
                'location': [location.x, location.y, location.z], 
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'intrinsic': K.tolist(),
                'world2cam': world2cam,
                'cam2ego': cam2ego.tolist(),
                'fov': fov,
                'image_size_x': width,
                'image_size_y': height,
            }
            results[key] = result

        # radar
        for value in world.get_actors().filter('*sensor.other.radar*'):
            sensors[value.attributes['role_name']] = value

        for key in ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']:
            value = sensors[key]
            location = value.get_transform().location
            rotation = value.get_transform().rotation
            world2radar = value.get_transform().get_inverse_matrix()
            radar2ego = get_matrix(location=[self.sensors_mapping[key]['x'], self.sensors_mapping[key]['y'], self.sensors_mapping[key]['z']], 
                        rotation=[self.sensors_mapping[key]['pitch'], self.sensors_mapping[key]['roll'], self.sensors_mapping[key]['yaw']])
            result ={
                'location': [location.x, location.y, location.z], 
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'world2radar': world2radar,
                'radar2ego': radar2ego.tolist(),
            }
            results[key] = result
        
        # lidar
        for value in world.get_actors().filter('*sensor.lidar.ray_cast*'):
            sensors[value.attributes['role_name']] = value

        for key in ['LIDAR_TOP']:
            value = sensors[key]
            location = value.get_transform().location
            rotation = value.get_transform().rotation
            world2lidar = value.get_transform().get_inverse_matrix()
            lidar2ego = get_matrix(location=[self.sensors_mapping[key]['x'], self.sensors_mapping[key]['y'], self.sensors_mapping[key]['z']], 
                        rotation=[self.sensors_mapping[key]['pitch'], self.sensors_mapping[key]['roll'], self.sensors_mapping[key]['yaw']])
            result ={
                'location': [location.x, location.y, location.z], 
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'world2lidar': world2lidar,
                'lidar2ego': lidar2ego.tolist(),
            }
            results[key] = result
        return results

    def get_bounding_boxes(self, lidar=None, radar=None):
        results = []

        # ego_vehicle
        npc = self._vehicle
        npc_id = str(npc.id)
        npc_type_id = npc.type_id
        npc_base_type = npc.attributes['base_type']
        location = npc.get_transform().location
        rotation = npc.get_transform().rotation
        # 
        # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
        # center, extent = get_center_and_extent(verts)
        # from carla official
        # bb_cords = _bounding_box_to_world(npc.bounding_box)
        # world_cord = _vehicle_to_world(bb_cords, npc)
        # from handcraft
        extent = npc.bounding_box.extent
        center = npc.get_transform().transform(npc.bounding_box.location)
        local_verts = calculate_cube_vertices(npc.bounding_box.location, npc.bounding_box.extent)
        global_verts = []
        for l_v in local_verts:
            g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
            global_verts.append([g_v.x, g_v.y, g_v.z])
        ###################
        ego_speed = self._get_forward_speed(transform=npc.get_transform(), velocity=npc.get_velocity())
        ego_brake = npc.get_control().brake
        ego_matrix = np.array(npc.get_transform().get_matrix())
        ego_yaw = np.deg2rad(rotation.yaw)
        road_id = self.world_map.get_waypoint(location).road_id
        lane_id = self.world_map.get_waypoint(location).lane_id
        section_id = self.world_map.get_waypoint(location).section_id
        world2ego = npc.get_transform().get_inverse_matrix()

        result = {
            'class': 'ego_vehicle',
            'id': npc_id,
            'type_id': npc_type_id,
            'base_type': npc_base_type,
            'location': [location.x, location.y, location.z],
            'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
            'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
            'center': [center.x, center.y, center.z],
            'extent': [extent.x, extent.y, extent.z],
            'world_cord': global_verts,
            'semantic_tags': [npc.semantic_tags],
            'color': npc.attributes['color'],
            'speed': ego_speed,
            'brake': ego_brake,
            'road_id': road_id,
            'lane_id': lane_id,
            'section_id': section_id,
            'world2ego': world2ego,
        }
        results.append(result)

        # vehicles.vehicle
        vehicles = self.get_actor_filter_vehicle()
        for npc in vehicles.vehicle:
            if not npc.is_alive: continue
            if npc.id == self._vehicle.id: continue
            npc_id = str(npc.id)
            location = npc.get_transform().location
            rotation = npc.get_transform().rotation
            road_id = self.world_map.get_waypoint(location).road_id
            lane_id = self.world_map.get_waypoint(location).lane_id
            section_id = self.world_map.get_waypoint(location).section_id
            # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
            # center, extent = get_center_and_extent(verts)
            # # from carla official
            # bb_cords = _bounding_box_to_world(npc.bounding_box)
            # world_cord = _vehicle_to_world(bb_cords, npc)
            # #
            # from handcraft
            world2vehicle = npc.get_transform().get_inverse_matrix()
            extent = npc.bounding_box.extent
            center = npc.get_transform().transform(npc.bounding_box.location)
            local_verts = calculate_cube_vertices(npc.bounding_box.location, npc.bounding_box.extent)
            global_verts = []
            for l_v in local_verts:
                g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
                global_verts.append([g_v.x, g_v.y, g_v.z])
            ###################
            vehicle_speed = self._get_forward_speed(transform=npc.get_transform(), velocity=npc.get_velocity())
            vehicle_brake = npc.get_control().brake
            vehicle_matrix = np.array(npc.get_transform().get_matrix())
            yaw = np.deg2rad(rotation.yaw)
            try:
                light_state = str(npc.get_light_state()).split('.')[-1]
            except:
                light_state = 'None'
            # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = get_relative_transform(ego_matrix, vehicle_matrix)
            if not lidar is None:
                num_in_bbox_lidar_points = self.get_lidar_points_in_bbox(relative_pos, relative_yaw, extent, lidar)
            else:
                num_in_bbox_lidar_points = -1
            
            distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
            result = {
                'class': 'vehicle',
                'state': 'dynamic',
                'id': npc_id,
                'location': [location.x, location.y, location.z],
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
                'center': [center.x, center.y, center.z],
                'extent': [extent.x, extent.y, extent.z],
                'world_cord': global_verts,
                'semantic_tags': npc.semantic_tags,
                'type_id': npc.type_id,
                'color': npc.attributes['color'],
                'base_type': npc.attributes['base_type'],
                'num_points': int(num_in_bbox_lidar_points),
                'distance': distance,
                'speed': vehicle_speed,
                'brake': vehicle_brake,
                'light_state': light_state,
                'road_id': road_id,
                'lane_id': lane_id,
                'section_id': section_id,
                'world2vehicle': world2vehicle,
                # 'actor': npc, # for debug 
              }
            results.append(result)
        
        world = CarlaDataProvider.get_world()

        # vehicles.static
        car_bbox_list = world.get_level_bbs(carla.CityObjectLabel.Car) 
        bicycle_list = world.get_level_bbs(carla.CityObjectLabel.Bicycle)
        bus_list = world.get_level_bbs(carla.CityObjectLabel.Bus)
        motorcycle_list = world.get_level_bbs(carla.CityObjectLabel.Motorcycle)
        train_list = world.get_level_bbs(carla.CityObjectLabel.Train)
        truck_list = world.get_level_bbs(carla.CityObjectLabel.Truck)
        vehicles_static_bbox = car_bbox_list + bicycle_list + bus_list + motorcycle_list + train_list + truck_list
        vehicles_static_bbox_nearby = []
        for v_s in vehicles_static_bbox:
            if compute_2d_distance(v_s.location, self._vehicle.get_transform().location) < (DIS_LIGHT_SAVE + 20):
                vehicles_static_bbox_nearby.append(v_s)
        for npc in vehicles.static:
            if not npc.is_alive: continue
            new_bbox = None
            min_dis = 50
            for vehicle_bbox in vehicles_static_bbox_nearby:
                dis = compute_2d_distance(npc.get_transform().location, vehicle_bbox.location)
                if dis < min_dis:
                    new_bbox = vehicle_bbox
                    min_dis = dis
            if min_dis > 20:
                continue
            if not new_bbox:
                raise Exception('new_bbox is None')
            if new_bbox not in vehicles_static_bbox_nearby:
                raise Exception('new_bbox not in vehicles_static_bbox_nearby')
            vehicles_static_bbox_nearby.remove(new_bbox)
            npc_id = str(npc.id)
            # location = new_bbox.location
            # rotation = new_bbox.rotation
            # center = new_bbox.location
            extent = new_bbox.extent
            ####
            location = npc.get_transform().location
            rotation = npc.get_transform().rotation
            road_id = self.world_map.get_waypoint(location).road_id
            lane_id = self.world_map.get_waypoint(location).lane_id
            section_id = self.world_map.get_waypoint(location).section_id
            # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
            # center, extent = get_center_and_extent(verts)
            # # from carla official
            # bb_cords = _bounding_box_to_world(npc.bounding_box)
            # world_cord = _vehicle_to_world(bb_cords, npc)
            # #
            # from handcraft
            world2vehicle = npc.get_transform().get_inverse_matrix()
            # extent = npc.bounding_box.extent
            # extent = carla.Vector3D(extent.y, extent.x, extent.z) # staic need swap
            center = npc.get_transform().transform(npc.bounding_box.location)
            local_verts = calculate_cube_vertices(npc.bounding_box.location, extent)
            global_verts = []
            for l_v in local_verts:
                g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
                # g_v = np.dot(np.matrix(npc.get_transform().get_inverse_matrix()).I, [l_v[0], l_v[1], l_v[2],1])
                global_verts.append([g_v.x, g_v.y, g_v.z])
            ###################
            vehicle_speed = self._get_forward_speed(transform=npc.get_transform(), velocity=npc.get_velocity())
            vehicle_brake = 1.0
            vehicle_matrix = np.array(npc.get_transform().get_matrix())
            yaw = np.deg2rad(rotation.yaw)
            light_state= 'NONE'
            # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = get_relative_transform(ego_matrix, vehicle_matrix)
            if not lidar is None:
                num_in_bbox_points = self.get_lidar_points_in_bbox(relative_pos, relative_yaw, extent, lidar)
            else:
                num_in_bbox_points = -1

            distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
            result = {
                'class': 'vehicle',
                'state': 'static',
                'id': npc_id,
                'location': [location.x, location.y, location.z],
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
                'center': [center.x, center.y, center.z],
                'extent': [extent.x, extent.y, extent.z],
                'world_cord': global_verts,
                'semantic_tags': npc.semantic_tags,
                'type_id': npc.attributes['mesh_path'],
                'num_points': int(num_in_bbox_points),
                'distance': distance,
                'speed': vehicle_speed,
                'brake': vehicle_brake,
                'light_state': light_state,
                'road_id': road_id,
                'lane_id': lane_id,
                'section_id': section_id,
                'world2vehicle': world2vehicle,
                # 'actor': npc, # for debug
                }
            results.append(result)
        
        # pedestrians
        pedestrians = world.get_actors().filter('walker*')
        for npc in pedestrians:
            if not npc.is_alive: 
                continue
            else:
                try:
                    if compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location) < DIS_WALKER_SAVE:
                        npc_id = str(npc.id)
                        location = npc.get_transform().location
                        rotation = npc.get_transform().rotation
                        road_id = self.world_map.get_waypoint(location).road_id
                        lane_id = self.world_map.get_waypoint(location).lane_id
                        section_id = self.world_map.get_waypoint(location).section_id
                        # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
                        # center, extent = get_center_and_extent(verts)
                        # # from carla official
                        # bb_cords = _bounding_box_to_world(npc.bounding_box)
                        # world_cord = _vehicle_to_world(bb_cords, npc)
                        # #
                        # from handcraft
                        world2ped = npc.get_transform().get_inverse_matrix()
                        extent = npc.bounding_box.extent
                        center = npc.get_transform().transform(npc.bounding_box.location)
                        local_verts = calculate_cube_vertices(npc.bounding_box.location, npc.bounding_box.extent)
                        global_verts = []
                        for l_v in local_verts:
                            g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
                            global_verts.append([g_v.x, g_v.y, g_v.z])
                        ###################
                        walker_speed = self._get_forward_speed(transform=npc.get_transform(), velocity=npc.get_velocity())
                        # walker_speed = npc.attributes['speed'] #(TODO) yzj
                        walker_matrix = np.array(npc.get_transform().get_matrix())
                        yaw = np.deg2rad(rotation.yaw)
                        bones_3d_lines = build_skeleton(npc, self.skeleton_links)
                        # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                        relative_yaw = normalize_angle(yaw - ego_yaw)
                        relative_pos = get_relative_transform(ego_matrix, walker_matrix)
                        if not lidar is None:
                            num_in_bbox_points = self.get_lidar_points_in_bbox(relative_pos, relative_yaw, extent, lidar)
                        else:
                            num_in_bbox_points = -1

                        distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
                        result = {
                            'class': 'walker',
                            'id': npc_id,
                            'location': [location.x, location.y, location.z],
                            'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                            'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
                            'center': [center.x, center.y, center.z],
                            'extent': [extent.x, extent.y, extent.z],
                            'world_cord': global_verts,
                            'semantic_tags': npc.semantic_tags,
                            'type_id': npc.type_id,
                            'gender': npc.attributes['gender'],
                            'age': npc.attributes['age'],
                            'num_points': int(num_in_bbox_points),
                            'distance': distance,
                            'speed': walker_speed,
                            'bone': bones_3d_lines,
                            'road_id': road_id,
                            'lane_id': lane_id,
                            'section_id': section_id,
                            'world2ped': world2ped,
                            # 'actor': npc, # for debug
                        }
                        results.append(result)
                except RuntimeError:
                    continue
        
        # traffic_light
        traffic_light = self.get_actor_filter_traffic_light()
        traffic_light_bbox = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        traffic_light_bbox_nearby = []
        for light_bbox in traffic_light_bbox:
            if compute_2d_distance(light_bbox.location, self._vehicle.get_transform().location) < (DIS_LIGHT_SAVE + 20):
                traffic_light_bbox_nearby.append(light_bbox)
        for npc in traffic_light.lights:
            new_bbox = None
            min_dis = 50
            for light_bbox in traffic_light_bbox_nearby:
                dis = compute_2d_distance(npc.get_transform().location, light_bbox.location)
                if dis < min_dis:
                    new_bbox = light_bbox
                    min_dis = dis
            if min_dis > 20:
                continue
            traffic_light_bbox_nearby.remove(new_bbox)
            npc_id = str(npc.id)
            location = new_bbox.location
            rotation = new_bbox.rotation
            center = new_bbox.location
            extent = new_bbox.extent
            road_id = self.world_map.get_waypoint(new_bbox.location).road_id
            lane_id = self.world_map.get_waypoint(new_bbox.location).lane_id
            section_id = self.world_map.get_waypoint(new_bbox.location).section_id
            volume_location = npc.get_transform().transform(npc.trigger_volume.location)
            volume_rotation = carla.Rotation(pitch=(rotation.pitch + npc.trigger_volume.rotation.pitch)%360, roll=(rotation.roll + npc.trigger_volume.rotation.roll)%360, yaw=(rotation.yaw + npc.trigger_volume.rotation.yaw) % 360)
            state = npc.state
            distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
            if traffic_light.most_affect_light and str(traffic_light.most_affect_light.id) == npc_id:
                affects_ego = True
            else:
                affects_ego = False
            result = {
                'class': 'traffic_light',
                'id': npc_id,
                'location': [location.x, location.y, location.z],
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'center': [center.x, center.y, center.z],
                'extent': [extent.x, extent.y, extent.z],
                'semantic_tags': npc.semantic_tags,
                'type_id': npc.type_id,
                'distance': distance,
                'state': state,
                'affects_ego': affects_ego,
                'trigger_volume_location': [volume_location.x, volume_location.y, volume_location.z],
                'trigger_volume_rotation': [volume_rotation.pitch, volume_rotation.roll, volume_rotation.yaw],
                'trigger_volume_extent': [npc.trigger_volume.extent.x, npc.trigger_volume.extent.y, npc.trigger_volume.extent.z],
                'road_id': road_id,
                'lane_id': lane_id,
                'section_id': section_id,
                # 'actor': npc, # for debug
                # 'new_bbox': new_bbox, # for debug
            }
            results.append(result)
        
        # traffic_sign
        traffic_sign = self.get_actor_filter_traffic_sign()
        traffic_sign_bbox = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        traffic_sign_bbox_nearby = []
        for sign_bbox in traffic_sign_bbox:
            if compute_2d_distance(sign_bbox.location, self._vehicle.get_transform().location) < (DIS_SIGN_SAVE + 20):
                traffic_sign_bbox_nearby.append(sign_bbox)

        for npc in traffic_sign.actors:
            if hasattr(npc, 'trigger_volume'):
                new_bbox = None
                min_dis = 50
                for sign_bbox in traffic_sign_bbox_nearby:
                    dis = compute_2d_distance(npc.get_transform().location, sign_bbox.location)
                    if dis < min_dis:
                        new_bbox = sign_bbox
                        min_dis = dis
                if min_dis > 20:
                    continue
                traffic_sign_bbox_nearby.remove(new_bbox)
                npc_id = str(npc.id)
                world2sign = npc.get_transform().get_inverse_matrix()
                location = new_bbox.location
                rotation = new_bbox.rotation
                center = new_bbox.location
                extent = new_bbox.extent
                road_id = self.world_map.get_waypoint(new_bbox.location).road_id
                lane_id = self.world_map.get_waypoint(new_bbox.location).lane_id
                section_id = self.world_map.get_waypoint(new_bbox.location).section_id
                volume_location = npc.get_transform().transform(npc.trigger_volume.location)
                volume_rotation = carla.Rotation(pitch=(rotation.pitch + npc.trigger_volume.rotation.pitch)%360, roll=(rotation.roll + npc.trigger_volume.rotation.roll)%360, yaw=(rotation.yaw + npc.trigger_volume.rotation.yaw) % 360)
                distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
                if traffic_sign.most_affect_sign and str(traffic_sign.most_affect_sign.id) == npc_id:
                    affects_ego = True
                else:
                    affects_ego = False
                result = {
                    'class': 'traffic_sign',
                    'id': npc_id,
                    'location': [location.x, location.y, location.z],
                    'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                    'center': [center.x, center.y, center.z],
                    'extent': [extent.x, extent.y, extent.z],
                    # 'extent': [extent.x, 0.5, extent.z],
                    'semantic_tags': npc.semantic_tags,
                    'type_id': npc.type_id,
                    'distance': distance,
                    'affects_ego': affects_ego,
                    'trigger_volume_location': [volume_location.x, volume_location.y, volume_location.z],
                    'trigger_volume_rotation': [volume_rotation.pitch, volume_rotation.roll, volume_rotation.yaw],
                    'trigger_volume_extent': [npc.trigger_volume.extent.x, npc.trigger_volume.extent.y, npc.trigger_volume.extent.z],
                    'road_id': road_id,
                    'lane_id': lane_id,
                    'section_id': section_id,
                    'world2sign': world2sign,
                    # 'actor': npc, # for debug
                    # 'new_bbox': new_bbox, # for debug
                }
            else:
                npc_id = str(npc.id)
                location = npc.get_transform().location
                rotation = npc.get_transform().rotation
                # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
                # center, extent = get_center_and_extent(verts)
                # from handcraft
                world2sign = npc.get_transform().get_inverse_matrix()
                extent = npc.bounding_box.extent
                center = npc.get_transform().transform(npc.bounding_box.location)
                local_verts = calculate_cube_vertices(npc.bounding_box.location, npc.bounding_box.extent)
                global_verts = []
                for l_v in local_verts:
                    g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
                    global_verts.append([g_v.x, g_v.y, g_v.z])
                road_id = self.world_map.get_waypoint(location).road_id
                lane_id = self.world_map.get_waypoint(location).lane_id
                section_id = self.world_map.get_waypoint(location).section_id
                # distance = npc.get_transform().location.distance(self._vehicle.get_transform().location)
                distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
                if traffic_sign.most_affect_sign and str(traffic_sign.most_affect_sign.id) == npc_id:
                    affects_ego = True
                else:
                    affects_ego = False

                result = {
                    'class': 'traffic_sign',
                    'id': npc_id,
                    'location': [location.x, location.y, location.z],
                    'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                    'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
                    'center': [center.x, center.y, center.z],
                    'extent': [extent.x, extent.y, extent.z],
                    'world_cord': global_verts,
                    # 'extent': [extent.x, 0.5, extent.z],
                    'semantic_tags': npc.semantic_tags,
                    'type_id': npc.type_id,
                    'distance': distance,
                    'affects_ego': affects_ego,
                    'road_id': road_id,
                    'lane_id': lane_id,
                    'section_id': section_id,
                    'world2sign': world2sign,
                    # 'actor': npc, # for debug
                }
            results.append(result)
        return results

    def polar_to_cartesian(self, altitude, azimuth, depth):
        """
        Convert polar coordinates (altitude, azimuth, depth) to Cartesian (x, y, z).
        Altitude and azimuth are assumed to be in radians.
        """
        z = depth * np.sin(altitude)
        r_cos_altitude = depth * np.cos(altitude)
        x = r_cos_altitude * np.cos(azimuth)
        y = r_cos_altitude * np.sin(azimuth)
        return x, y, z

    def get_radar_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, radar_data):
        """
        Checks for a given vehicle in ego coordinate system, how many RADAR hits there are in its bounding box.
        :param vehicle_pos: Relative position of the vehicle w.r.t. the ego [x, y, z]
        :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego in radians
        :param extent: List, half extent of the bounding box [length/2, width/2, height/2]
        :param radar_data: RADAR data with structure [altitude, azimuth, depth, velocity]
        :return: Returns the number of RADAR hits within the bounding box of the vehicle
        """
        radar_cartesian = np.array([self.polar_to_cartesian(np.radians(altitude), np.radians(azimuth), depth)
                                    for altitude, azimuth, depth, _ in radar_data])

        rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                    [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0],
                                    [0.0, 0.0, 1.0]])

        # Transform RADAR points to vehicle coordinate system
        vehicle_radar = (rotation_matrix.T @ (radar_cartesian - vehicle_pos).T).T

        # Half extents for the bounding box
        x, y, z = extent
        num_hits = ((vehicle_radar[:, 0] <= x) & (vehicle_radar[:, 0] >= -x) & 
                    (vehicle_radar[:, 1] <= y) & (vehicle_radar[:, 1] >= -y) & 
                    (vehicle_radar[:, 2] <= z) & (vehicle_radar[:, 2] >= -z)).sum()
        return num_hits
    
    def get_lidar_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, lidar):
        """
        Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
        :param vehicle_pos: Relative position of the vehicle w.r.t. the ego
        :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego
        :param extent: List, Extent of the bounding box
        :param lidar: LiDAR point cloud
        :return: Returns the number of LiDAR hits within the bounding box of the
        vehicle
        """

        rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                    [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

        # LiDAR in the with the vehicle as origin
        vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

        # check points in bbox
        x, y, z = extent.x, extent.y, extent.z
        num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                    (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
        return num_points
    
    def gps_to_location(self, gps):
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
        pass
    
    def _get_latlon_ref(self):
        """
        Convert from waypoints world coordinates to CARLA GPS coordinates
        :return: tuple with lat and lon coordinates
        """
        xodr = self.world_map.to_opendrive()
        tree = ET.ElementTree(ET.fromstring(xodr))

        # default reference
        lat_ref = 42.0
        lon_ref = 2.0

        for opendrive in tree.iter("OpenDRIVE"):
            for header in opendrive.iter("header"):
                for georef in header.iter("geoReference"):
                    if georef.text:
                        str_list = georef.text.split(' ')
                        for item in str_list:
                            if '+lat_0' in item:
                                lat_ref = float(item.split('=')[1])
                            if '+lon_0' in item:
                                lon_ref = float(item.split('=')[1])
        return lat_ref, lon_ref
