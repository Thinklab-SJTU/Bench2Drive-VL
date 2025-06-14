"""
Privileged driving agent used for data collection.
Drives by accessing the simulator directly.
"""

import os
import ujson
import datetime
import pathlib
import gzip
from collections import deque
from agents.navigation.local_planner import RoadOption
import math
import numpy as np
import carla
import json
from scipy.integrate import RK45

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents import autonomous_agent, autonomous_agent_local
from nav_planner import RoutePlanner
from lateral_controller import LateralPIDController
from privileged_route_planner import PrivilegedRoutePlanner
from config import GlobalConfig
import transfuser_utils as t_u
from scenario_logger import ScenarioLogger
from longitudinal_controller import LongitudinalLinearRegressionController
from kinematic_bicycle_model import KinematicBicycleModel

# B2DVL-Adapter
from qa_process import find_qdict_by_id
from evaluator import extract_keys, extract_key_list
from generator_modules import SpeedCommand, DirectionCommand
from generator_modules import get_lane_deviate_info, is_on_road
from io_utils import print_warning, print_error, print_green, print_bottom, write_json, print_debug

MINIMAL = int(os.environ.get("MINIMAL", 0))
if MINIMAL <= 0:
    from carla_inference import get_carla_inference_worker

def get_entry_point():
    return "AutoPilot"

class AutoPilot(autonomous_agent_local.AutonomousAgent):
    """
      Privileged driving agent used for data collection.
      Drives by accessing the simulator directly.
      """

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        """
        Set up the autonomous agent for the CARLA simulation.

        Args:
            config_file_path (str): Path to the configuration file.
            route_index (int, optional): Index of the route to follow.
            traffic_manager (object, optional): The traffic manager object.

        """
        self.recording = False
        # self.track = autonomous_agent.Track.SENSORS
        self.track = autonomous_agent.Track.MAP
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False
        self.save_path = None
        self.route_index = route_index

        self.datagen = int(os.environ.get("VQA_GEN", 0)) == 1

        print_debug(f"[debug] _vlm_cfg_dir = {self._vlm_cfg_dir}")
        self.config = GlobalConfig(self._vlm_cfg_dir)

        self.speed_histogram = []
        self.make_histogram = int(os.environ.get("HISTOGRAM", 0))

        self.tp_stats = False
        self.tp_sign_agrees_with_angle = []
        if int(os.environ.get("TP_STATS", 0)):
            self.tp_stats = True

        # Dynamics models
        self.ego_model = KinematicBicycleModel(self.config)
        self.vehicle_model = KinematicBicycleModel(self.config)

        # Configuration
        self.visualize = int(os.environ.get("DEBUG_CHALLENGE", 0))

        self.walker_close = False
        self.distance_to_walker = np.inf
        self.stop_sign_close = False

        # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
        self.ego_blocked_for_ticks = 0

        # Controllers
        self._turn_controller = LateralPIDController(self.config)

        self.list_traffic_lights = []

        # Navigation command buffer, needed because the correct command comes from the last cleared waypoint
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.next_commands = deque(maxlen=2)
        self.next_commands.append(4)
        self.next_commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # Initialize controls
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.target_speed = self.config.target_speed_fast

        self.augmentation_translation = 0
        self.augmentation_rotation = 0

        # Angle to the next waypoint, normalized in [-1, 1] corresponding to [-90, 90]
        self.angle = 0.0
        self.stop_sign_hazard = False
        self.traffic_light_hazard = False
        self.walker_hazard = False
        self.vehicle_hazard = False
        self.junction = False
        self.aim_wp = None  # Waypoint the expert is steering towards
        self.remaining_route = None  # Remaining route
        self.remaining_route_original = None  # Remaining original route
        self.route_waypoints = [] # added
        self.route_points = []
        self.close_traffic_lights = []
        self.close_stop_signs = []
        self.was_at_stop_sign = False
        self.cleared_stop_sign = False
        self.visible_walker_ids = []
        self.walker_past_pos = {}  # Position of walker in the last frame

        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # Get the world map and the ego vehicle
        self.world_map = CarlaDataProvider.get_map()
        self.town_name = self.world_map.name.split('/')[-1]
        self.last_measurement_data = None
        self.last_basic_info = []

        # Set up the save path if specified
        if os.environ.get("SAVE_PATH", None) is not None:
            now = datetime.datetime.now()
            # # Determine the scenario name and number
            # string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            # # print(f"[debug] weather name is {self.weather_name}")
            # if self.weather_name is not None:
            #     string += f"_{self.weather_name}"

            # string += "_".join(map(lambda x: f"{x:02}", (now.month, now.day, now.hour, now.minute, now.second)))

            string = self._save_name

            self.save_path = pathlib.Path(os.environ["SAVE_PATH"]) / self.config.folder_str / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            if self.datagen:
                (self.save_path / "measurements").mkdir()

            self.lon_logger = ScenarioLogger(
                save_path=self.save_path,
                route_index=route_index,
                logging_freq=self.config.logging_freq,
                log_only=True,
                route_only=False,  # with vehicles
                roi=self.config.logger_region_of_interest,
            )

        # B2DVL-Adapter
        # self.inference_worker = InferenceWorker(worker_id=0, scenario_list=[],
        #                                         dataset=None, transform=None,
        #                                         model=None, model_name="gt", model_path=None,
        #                                         outpath=None, configs=None, in_carla=True)
            

        self.model_name = self.config.model_name
        self.model_path = self.config.model_path
        if MINIMAL <= 0:
            self.inference_worker = get_carla_inference_worker(config_dir=self.config.vlm_config_dir, 
                                                               model_name=self.model_name,
                                                               model_path=self.model_path,
                                                               outpath=self.config.infer_res_dir,
                                                               gpu_id=self.config.vlm_gpu_id)
        
        self.all_spd_cmds = SpeedCommand()
        self.all_dir_cmds = DirectionCommand()
        self.last_command = None
        self.last_command_far = None

        self.last_dir_cmd = None
        self.last_spd_cmd = None

        self.leave_highway_scenarios = ['HighwayExit', 'MergerIntoSlowTraffic', 'InterurbanActorFlow']
        self.enter_highway_scenarios = ['HighwayCutIn', 'MergerIntoSlowTrafficV2', 'InterurbanAdvancedActorFlow']
        self.right_merging_scenarios = ['NonSignalizedJunctionRightTurn', 'SignalizedJunctionRightTurn']
        self.entered_junction_frame_count = 0

        self.front_camera_map = {
            'CAM_FRONT':            'rgb_front',
            'CAM_FRONT_LEFT':       'rgb_front_left',
            'CAM_FRONT_RIGHT':      'rgb_front_right',
            'CAM_BACK':             'rgb_back',
            'CAM_BACK_LEFT':        'rgb_back_left',
            'CAM_BACK_RIGHT':       'rgb_back_right',
            'ANNO_CAM_FRONT':       'anno_rgb_front'
        }

        self.concat_camera_map = {
            'CAM_FRONT':            'rgb_front',
            'CAM_FRONT_LEFT':       'rgb_front_left',
            'CAM_FRONT_RIGHT':      'rgb_front_right',
            'CAM_BACK':             'rgb_back',
            'CAM_BACK_LEFT':        'rgb_back_left',
            'CAM_BACK_RIGHT':       'rgb_back_right',
            'ANNO_CAM_FRONT':       'anno_rgb_front',
            'ANNO_CAM_FRONT_LEFT':  'anno_rgb_front_left',
            'ANNO_CAM_FRONT_RIGHT': 'anno_rgb_front_right',
            'ANNO_CAM_BACK':        'anno_rgb_back',
            'ANNO_CAM_BACK_LEFT':   'anno_rgb_back_left',
            'ANNO_CAM_BACK_RIGHT':  'anno_rgb_back_right',
            'CAM_FRONT_CONCAT':     'front_concat',
            'CAM_BACK_CONCAT':      'back_concat'
        }

        self.bev_camera_map = {
            'CAM_FRONT':            'rgb_front',
            'CAM_FRONT_LEFT':       'rgb_front_left',
            'CAM_FRONT_RIGHT':      'rgb_front_right',
            'CAM_BACK':             'rgb_back',
            'CAM_BACK_LEFT':        'rgb_back_left',
            'CAM_BACK_RIGHT':       'rgb_back_right',
            'BEV':                  'bev',
            'ANNO_BEV':             'anno_bev'
        }

    def toggle_recording(self, force_stop=False):
        """
        Toggle the recording of the simulation data.

        Args:
            force_stop (bool, optional): If True, stop the recording regardless of the current state.
        """
        # Toggle the recording state and determine the text
        self.recording = not self.recording

        if self.recording and not force_stop:
            self.client = CarlaDataProvider.get_client()

            # Determine the scenario name and number
            # scenario_name = pathlib.Path(self.config_path).parent.stem
            # scenario_number = pathlib.Path(self.config_path).stem
            scenario_name = self._curr_scenario_name
            scenario_number = self._curr_route_name

            # Construct the log file path
            log_path = f"{pathlib.Path(os.environ['SAVE_PATH'])}/{self.config.folder_str}/{scenario_name}/{scenario_number}.log"

            print(f"Saving to {log_path}")
            pathlib.Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)

            # Start the recorder with the specified log path
            self.client.start_recorder(log_path, True)
        else:
            # Stop the recorder
            self.client.stop_recorder()

    def _init(self, hd_map):
        """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.

        Args:
            hd_map (carla.Map): The map object of the CARLA world.
        """
        print("Sparse Waypoints:", len(self._global_plan))
        print("Dense Waypoints:", len(self.org_dense_route_world_coord))

        # Get the hero vehicle and the CARLA world
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(self._vehicle.get_location())
        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2

        # Set up the route planner and extrapolation
        self._waypoint_planner = PrivilegedRoutePlanner(self.config)
        self._waypoint_planner.setup_route(self.org_dense_route_world_coord, self._world, self.world_map,
                                           starts_with_parking_exit, self._vehicle.get_location())
        self._waypoint_planner.save()

        # Set up the longitudinal controller and command planner
        self._longitudinal_controller = LongitudinalLinearRegressionController(self.config)
        self._command_planner = RoutePlanner(self.config.route_planner_min_distance,
                                             self.config.route_planner_max_distance)
        self._command_planner.set_route(self._global_plan_world_coord)

        # Set up logging
        if self.save_path is not None:
            self.lon_logger.ego_vehicle = self._vehicle
            self.lon_logger.world = self._world

        # Preprocess traffic lights
        all_actors = self._world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = t_u.get_traffic_light_waypoints(actor, self.world_map)
                self.list_traffic_lights.append((actor, center, waypoints))

        # Remove bugged 2-wheelers
        # https://github.com/carla-simulator/carla/issues/3670
        for actor in all_actors:
            if "vehicle" in actor.type_id:
                extent = actor.bounding_box.extent
                if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
                    actor.destroy()

        self.initialized = True

    def sensors(self):
        """
        Returns a list of sensor specifications for the ego vehicle.

        Each sensor specification is a dictionary containing the sensor type,
        reading frequency, position, and other relevant parameters.

        Returns:
            list: A list of sensor specification dictionaries.
        """
        sensor_specs = [{
            "type": "sensor.opendrive_map",
            "reading_frequency": 1e-6,
            "id": "hd_map"
        }, {
            "type": "sensor.other.imu",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sensor_tick": 0.05,
            "id": "imu"
        }, {
            "type": "sensor.speedometer",
            "reading_frequency": 20,
            "id": "speed"
        }]

        return sensor_specs

    def tick_autopilot(self, input_data):
        """
        Get the current state of the vehicle from the input data and the vehicle's sensors.

        Args:
            input_data (dict): Input data containing sensor information.

        Returns:
            dict: A dictionary containing the vehicle's position (GPS), speed, and compass heading.
        """
        # Get the vehicle's speed from its velocity vector
        speed = self._vehicle.get_velocity().length()

        # Get the IMU data from the input data
        imu_data = input_data["IMU"][1][-1]

        # Preprocess the compass data from the IMU
        compass = t_u.preprocess_compass(imu_data)

        # Get the vehicle's position from its location
        position = self._vehicle.get_location()
        gps = np.array([position.x, position.y, position.z])

        # Create a dictionary containing the vehicle's state
        vehicle_state = {
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        return vehicle_state

    def run_step(self, input_data, timestamp, sensors=None, vqa_data={}, plant=False):
        """
        Run a single step of the agent's control loop.

        Args:
            input_data (dict): Input data for the current step.
            timestamp (float): Timestamp of the current step.
            sensors (list, optional): List of sensor objects. Default is None.
            plant (bool, optional): Flag indicating whether to run the plant simulation or not. Default is False.

        Returns:
            If plant is False, it returns the control commands (steer, throttle, brake).
            If plant is True, it returns the driving data for the current step.
        """
        self.step += 1

        # Initialize the agent if not done yet
        if not self.initialized:
            client = CarlaDataProvider.get_client()
            world_map = client.get_world().get_map()
            self._init(world_map)

        # Get the control commands and driving data for the current step
        control, driving_data = self._get_control(input_data, vqa_data, plant)

        if plant:
            return driving_data
        else:
            return control

    def _get_control(self, input_data, vqa_data, plant):
        """
        Compute the control commands and save the driving data for the current frame.

        Args:
            input_data (dict): Input data for the current frame.
            plant (object): The plant object representing the vehicle dynamics.

        Returns:
            tuple: A tuple containing the control commands (steer, throttle, brake) and the driving data.
        """
        # vqa debug
        dir_cmd = None
        spd_cmd = None
        dir_cmd_gt = None
        spd_cmd_gt = None
        frame_number = self.step // self.config.data_save_freq
        frame_string = f'{(self.step // self.config.data_save_freq):05d}'
        # for key, value in vqa_data.items():
        #     print(f"[debug] key in vqa_data: {key}") #

        is_ready = True
        qdict50_list = None

        # Get the current speed and target speed
        tick_data = self.tick_autopilot(input_data)
        ego_position = tick_data["gps"]
        ego_speed = tick_data["speed"]
        location = self._vehicle.get_transform().location
        rotation = self._vehicle.get_transform().rotation
        ego_dict = {
            'location': [location.x, location.y, location.z],
            'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
        }

        ego_wp = self.world_map.get_waypoint(location)
        self.junction = ego_wp.is_junction
        lane_type = ego_wp.lane_type
        lane_type_str = str(ego_wp.lane_type)
        other_scene_info = CarlaDataProvider.get_scene_info('other_info')

        if 'QA' in vqa_data:
            command_int = None
            next_command_int = None
            try:
                command_int = self.commands[-2]
                next_command_int = self.next_commands[-2]
            except Exception as e:
                print_debug(f"[debug] fetching command_int and next_command_int in autopilot, error occured: {e}")
            self.last_command = command_int
            self.last_command_far = next_command_int
            print_debug(f"[debug] [self.last_command, self.last_command_far] = {[self.last_command, self.last_command_far]}")
            if self.junction:
                self.entered_junction_frame_count += 1
            else:
                self.entered_junction_frame_count = 0
            print_debug(f"[debug] entered_junction_frame_count = {self.entered_junction_frame_count}")
        
        action_desc_gt = None
        if 'QA' in vqa_data: # extract some gts
            qdict50_list = find_qdict_by_id(50, vqa_data)
            if qdict50_list is not None:
                dir_cmd_gt, spd_cmd_gt = extract_keys(qdict50_list[0]['A'])
            qdict43_list = find_qdict_by_id(43, vqa_data)
            if qdict43_list is not None:
                action_desc_gt = qdict43_list[0]['A']
        
        command_int = None
        next_command_int = None
        try:
            command_int = self.commands[-2]
            next_command_int = self.next_commands[-2]
        except:
            pass
        print_debug(f"[debug] [command_int, next_command_int] = {[command_int, next_command_int]}")

        # get command from vqagen and vlm
        if MINIMAL <= 0 and 'QA' in vqa_data:
            print_debug(f"[debug] all keys in vqa_data: {[x for x in vqa_data.keys()]}")
            image_dict = {}
            image_dict['frame_number'] = frame_number

            activated_map = {}
            if self.config.use_all_cams:
                activated_map = self.concat_camera_map
            if self.config.use_bev:
                activated_map = self.bev_camera_map
            if (not self.config.use_all_cams) and (not self.config.use_bev):
                activated_map = self.front_camera_map
            
            for camera_tag, camera_dir in activated_map.items():
                image_path = os.path.join(self.config.work_dir, self.save_path, 'camera', camera_dir, f"{frame_string}.jpg")
                if os.path.exists(image_path):
                    image_dict[camera_tag] = image_path
                else:
                    is_ready = False
            
            if is_ready:
                vqa_file = os.path.join(self.config.work_dir, self.config.output_graph_dir, self._save_name, f"{frame_number:05d}.json")
                print_debug(f'[debug] vqa_file_path = {vqa_file}')
                anno_file = os.path.join(self.config.work_dir, self.save_path, 'anno', f'{(self.step // self.config.data_save_freq):05d}.json.gz')
                print_debug(f'[debug] anno_path = {anno_file}')

                vqa_info = {
                    'scenario': self._save_name,
                    'frame_number': frame_number,
                    'frame_string': frame_string,
                    'file_path': vqa_file,
                    'content': vqa_data,
                    'anno': anno_file
                }

                special_state_str = None
                if action_desc_gt is not None and 'exit the parking space' in action_desc_gt:
                    if 'left' in action_desc_gt:
                        special_state_str = "change to the left lane to exit the parking space"
                    elif 'right' in action_desc_gt:
                        special_state_str = "change to the right lane to exit the parking space"             
                
                extra_prompt = f"Basic infos of the actors: {json.dumps(self.last_basic_info)}" if not self.config.no_perc_info else ""
                infer_res = self.inference_worker.ask_single_frame(scenario=self._save_name, 
                                                                   frame_number=frame_number, 
                                                                   vqa_data=vqa_info, 
                                                                   images=image_dict,
                                                                   entry=0, 
                                                                   start_index=0,
                                                                   end_index=-1,
                                                                   extra_prompt=extra_prompt,
                                                                   special_state_str=special_state_str)
                json_content, json_path = infer_res
                print_debug(f"[debug] json_path = {json_path}")
                write_json(json_content, json_path)

                qdict50_list = find_qdict_by_id(50, json_content)
                if qdict50_list is not None:
                    dir_cmd, spd_cmd = extract_key_list(qdict50_list[0]['VLM_answer'])
                    dir_cmd_gt, spd_cmd_gt = extract_keys(qdict50_list[0]['A'])
                    print_debug(f"[debug] doing VLM, VLM's answer: {qdict50_list[0]['VLM_answer']}, dir_cmd_list = {dir_cmd}, spd_cmd_list = {spd_cmd}")
        
        if MINIMAL and 'QA' in vqa_data: # use gt
            qdict50_list = find_qdict_by_id(50, vqa_data)
            if qdict50_list is not None:
                dir_cmd, spd_cmd = extract_keys(qdict50_list[0]['A'])

        if isinstance(dir_cmd, list) and len(dir_cmd) > 0:
            dir_cmd = dir_cmd[-1]
        if isinstance(spd_cmd, list) and len(spd_cmd) > 0:
            spd_cmd = spd_cmd[-1]
        if isinstance(dir_cmd_gt, list) and len(dir_cmd_gt) > 0:
            dir_cmd_gt = dir_cmd_gt[0]
        if isinstance(spd_cmd_gt, list) and len(spd_cmd_gt) > 0:
            spd_cmd_gt = spd_cmd_gt[0]
        
        if not self.junction and dir_cmd == self.all_dir_cmds.straight:
            dir_cmd = self.all_dir_cmds.follow
        
        if dir_cmd is not None: self.last_dir_cmd = dir_cmd
        if spd_cmd is not None: self.last_spd_cmd = spd_cmd
        if dir_cmd_gt is not None: self.last_dir_cmd_gt = dir_cmd_gt
        if spd_cmd_gt is not None: self.last_spd_cmd_gt = spd_cmd_gt
        
        if spd_cmd == None: spd_cmd = self.last_spd_cmd

        road_yaw_deg, ego_yaw_deg, deviant_degree = get_lane_deviate_info(self.world_map, ego_dict)
        ego_on_road = is_on_road(map=self.world_map, x=location.x, y=location.y, z=location.z)
        driving_opposite = abs(deviant_degree) > 90
        print_debug(f"[debug] _get_control scenario_name = {self._curr_scenario_name}, ego_speed = {ego_speed}, dir_cmd = {dir_cmd}, spd_cmd = {spd_cmd}, ego_on_road = {ego_on_road}, driving_opposite = {driving_opposite}")

        high_speed_scenario_flag = 'LaneChange' in self._curr_scenario_name or \
                                   self._curr_scenario_name in self.enter_highway_scenarios or \
                                   self._curr_scenario_name in self.leave_highway_scenarios
        
        scene_direction_value = None
        if isinstance(other_scene_info, dict):
            if 'direction' in other_scene_info and isinstance(other_scene_info['direction'], dict):
                if 'value' in other_scene_info['direction']:
                    scene_direction_value = other_scene_info['direction']['value']
        
        print_debug(f"[debug] autopilot, other_scene_info = {other_scene_info}, direction_value = {scene_direction_value}, lane_type = {lane_type}, lane_type_str = {lane_type_str}")
        
        UP_DELTA = 6.0
        DOWN_DELTA = 7.0
        BRAKE_MIN_SPEED = 2.0
        if high_speed_scenario_flag: BRAKE_MIN_SPEED = 3.0
        FOLLOW_LANE_TRANSITION_LENGTH = 40
        TRANSITION_LENGTH = 40
        if high_speed_scenario_flag:
            FOLLOW_LANE_TRANSITION_LENGTH *= 2
            TRANSITION_LENGTH *= 2
        
        CHANGE_LANE_TARGET_SPEED = 8.0
        # don't drive too fast when changing lane

        LANE_DEVIATE_FACTOR = 0.8 
        # deviation in InvadingTurningRoute unit: meter

        ENTER_JUNCTION_MAX_SPEED = 6.0 
        # when entering junction and drive too fast, the ego vehicle will crash into actor flow eveen if it wants to yield

        brake_flag = 0.0 # give to the handler below
        target_speed = ego_speed # initialized as 'keep'
        if spd_cmd == self.all_spd_cmds.accelerate:
            target_speed = ego_speed + UP_DELTA
        elif spd_cmd == self.all_spd_cmds.decelerate:
            target_speed = max(ego_speed - DOWN_DELTA, BRAKE_MIN_SPEED)
        elif spd_cmd == self.all_spd_cmds.stop:
            target_speed = 0.0
        else:
            # default: self.all_spd_cmds.keep:
            target_speed = ego_speed
        
        if self.junction and self._curr_scenario_name in self.right_merging_scenarios and \
           self.entered_junction_frame_count <= 3: # just entered the junction, lower the speed limit
            target_speed = min(target_speed, ENTER_JUNCTION_MAX_SPEED)

        if dir_cmd == self.all_dir_cmds.follow:
            self._waypoint_planner.follow_lane_at_location(location=location,
                                                           ego_current_wp=ego_wp,
                                                           transition_length=FOLLOW_LANE_TRANSITION_LENGTH,
                                                           lane_transition_factor=1)
        if dir_cmd == self.all_dir_cmds.left_change and \
           not ((scene_direction_value is not None and scene_direction_value == "right" and 'ParkingExit' in self._curr_scenario_name)):
            print_debug("[debug] autopilot: left change recognzed")
            self._waypoint_planner.change_lane_at_location(location=location,
                                                           ego_current_wp=ego_wp,
                                                           is_reverse=driving_opposite,
                                                           change_direction="left",
                                                           transition_length=TRANSITION_LENGTH,
                                                           lane_transition_factor=1,
                                                           ego_is_out_of_road=(not ego_on_road))
        if dir_cmd == self.all_dir_cmds.right_change and \
           not ((scene_direction_value is not None and scene_direction_value == "left" and 'ParkingExit' in self._curr_scenario_name)):
            print_debug("[debug] autopilot: right change recognzed")
            self._waypoint_planner.change_lane_at_location(location=location,
                                                           ego_current_wp=ego_wp,
                                                           is_reverse=driving_opposite,
                                                           change_direction="right",
                                                           transition_length=TRANSITION_LENGTH,
                                                           lane_transition_factor=1,
                                                           ego_is_out_of_road=(not ego_on_road))
        if dir_cmd == self.all_dir_cmds.left_deviate:
            print_debug("[debug] autopilot: left deviate recognzed")
            self._waypoint_planner.follow_lane_at_location(location=location,
                                                           ego_current_wp=ego_wp,
                                                           transition_length=TRANSITION_LENGTH,
                                                           lane_transition_factor=1.,
                                                           is_reversed=driving_opposite,
                                                           offset=-LANE_DEVIATE_FACTOR)
        if dir_cmd == self.all_dir_cmds.right_deviate:
            print_debug("[debug] autopilot: right deviate recognzed")
            self._waypoint_planner.follow_lane_at_location(location=location,
                                                           ego_current_wp=ego_wp,
                                                           transition_length=TRANSITION_LENGTH,
                                                           lane_transition_factor=1.,
                                                           is_reversed=driving_opposite,
                                                           offset=LANE_DEVIATE_FACTOR)

        if spd_cmd in [self.all_spd_cmds.accelerate, self.all_spd_cmds.keep] and \
           self.last_dir_cmd in [self.all_dir_cmds.left_change, self.all_dir_cmds.right_change] and \
           (not high_speed_scenario_flag):
            target_speed = CHANGE_LANE_TARGET_SPEED # don't change lane too fast on normal roads!


        # Get the list of vehicles in the scene
        actors = self._world.get_actors()
        vehicles = list(actors.filter("*vehicle*"))

        # Waypoint planning and route generation
        route_np, route_wp, _, distance_to_next_traffic_light, next_traffic_light, distance_to_next_stop_sign,\
                                        next_stop_sign, speed_limit = self._waypoint_planner.run_step(ego_position)
        self.route_waypoints = route_wp
        self.route_points = route_np

        # Extract relevant route information
        self.remaining_route = route_np[self.config.tf_first_checkpoint_distance:][::self.config.points_per_meter]
        self.remaining_route_original = self._waypoint_planner.original_route_points[
            self._waypoint_planner.route_index:][self.config.tf_first_checkpoint_distance:][::self.config.
                                                                                            points_per_meter]

        # Compute throttle and brake control
        throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(brake_flag, target_speed, ego_speed)

        # Compute steering control
        steer = self._get_steer(route_np, ego_position, tick_data["compass"], ego_speed)

        # Get steer by command
        if self.last_dir_cmd is not None and command_int is not None and self.last_command is not None and \
           (self.last_command == command_int): # command situation haven't changed
            if self.last_dir_cmd == self.all_dir_cmds.straight and \
               3 not in [self.last_command, self.last_command_far]:
                print_debug("[debug] setting steer to 0 because command is STRAIGHT")
                steer = 0
            if self.last_dir_cmd == self.all_dir_cmds.left_turn and \
               1 not in [self.last_command, self.last_command_far]:
                print_debug("[debug] setting steer to -1 because command is TURN_LEFT")
                steer = -1
            if self.last_dir_cmd == self.all_dir_cmds.right_turn and \
               2 not in [self.last_command, self.last_command_far]:
                print_debug("[debug] setting steer to 1 because command is TURN_RIGHT")
                steer = 1

        # Create the control command
        control = carla.VehicleControl()
        control.steer = steer + self.config.steer_noise * np.random.randn()
        control.throttle = throttle
        control.brake = max(brake_flag, control_brake)

        # print(f"[debug] final control: target_speed = {target_speed}, steer = {control.steer}, throttle = {control.throttle}, brake = {control.brake}")

        # # for ood [debug]
        # if self.junction:
        #     control.steer = np.random.uniform(-1, 0)
        #     control.throttle = 1
        #     control.brake = 0

        # Apply brake if the vehicle is stopped to prevent rolling back
        if control.throttle == 0 and ego_speed < self.config.minimum_speed_to_prevent_rolling_back:
            control.brake = 1

        # Apply throttle if the vehicle is blocked for too long
        ego_velocity = CarlaDataProvider.get_velocity(self._vehicle)
        if ego_velocity < 0.1:
            self.ego_blocked_for_ticks += 1
        else:
            self.ego_blocked_for_ticks = 0

        move_to_prevent_block = False
        if self.ego_blocked_for_ticks >= self.config.max_blocked_ticks:
            control.throttle = 1
            control.brake = 0
            move_to_prevent_block = True
        
        print_debug(f"[debug] before adjustment, control: target_speed = {target_speed}, steer = {control.steer}, throttle = {control.throttle}, brake = {control.brake}")
        print_debug(f"[debug] self.junction = {self.junction}, self.last_dir_cmd = {self.last_dir_cmd}, control.brake = {control.brake}")

        # if self.junction and \
        #    (self.last_dir_cmd == self.all_dir_cmds.left_turn or \
        #     self.last_dir_cmd == self.all_dir_cmds.right_turn) and \
        #    control.brake > 0.99:
        #     print_debug(f"[debug] modified steer for merging")
        #     if control.steer < 0:
        #         control.steer = max(control.steer / 3.0, -0.02)
        #     if control.steer > 0:
        #         control.steer = min(control.steer / 3.0, 0.02)
        #     # try to avoid avoid uneffective brake
        if self.junction and \
           (self.last_dir_cmd == self.all_dir_cmds.left_turn) and \
           control.brake > 0.99: # right turn needs curve
            print_debug(f"[debug] modified steer for merging")
            if control.steer < 0:
                control.steer = max(control.steer / 3.0, -0.02)
            if control.steer > 0:
                control.steer = min(control.steer / 3.0, 0.02)
            # try to avoid avoid uneffective brake
        
        if self.junction and move_to_prevent_block:
            print_debug(f"[debug] modified steer for unblocking")
            if control.steer < 0:
                control.steer = max(control.steer / 3.0, -0.2)
            if control.steer > 0:
                control.steer = min(control.steer / 3.0, 0.2)

        # Save control commands and target speed
        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        self.target_speed = target_speed

        # debug 2
        print_debug(f"[debug] final control: target_speed = {target_speed}, steer = {control.steer}, throttle = {control.throttle}, brake = {control.brake}")

        # Update speed histogram if enabled
        if self.make_histogram:
            self.speed_histogram.append((self.target_speed * 3.6) if not self.brake else 0.0)

        # Get the target and next target points from the command planner
        command_route = self._command_planner.run_step(ego_position)
        if len(command_route) > 2:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[2]
        elif len(command_route) > 1:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[1]
        else:
            target_point, far_command = command_route[0]
            next_target_point, next_far_command = command_route[0]

        # Update command history and save driving datas
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)
            self.next_commands.append(next_far_command.value)

        driving_data = self.save(target_point, next_target_point, steer, throttle, self.brake, control_brake, target_speed,
                                 speed_limit, tick_data)

        return control, driving_data

    def save(self, target_point, next_target_point, steering, throttle, brake, control_brake, target_speed, speed_limit,
             tick_data):
        """
        Save the driving data for the current frame.

        Args:
            target_point (numpy.ndarray): Coordinates of the target point.
            next_target_point (numpy.ndarray): Coordinates of the next target point.
            steering (float): The steering angle for the current frame.
            throttle (float): The throttle value for the current frame.
            brake (float): The brake value for the current frame.
            control_brake (bool): Whether the brake is controlled by the agent or not.
            target_speed (float): The target speed for the current frame.
            speed_limit (float): The speed limit for the current frame.
            tick_data (dict): Dictionary containing the current state of the vehicle.
            speed_reduced_by_obj (tuple): Tuple containing information about the object that caused speed reduction.

        Returns:
            dict: A dictionary containing the driving data for the current frame.
        """
        frame = self.step // self.config.data_save_freq

        # Extract relevant data from inputs
        target_point_2d = target_point[:2]
        next_target_point_2d = next_target_point[:2]
        ego_position = tick_data["gps"][:2]
        ego_orientation = tick_data["compass"]
        ego_speed = tick_data["speed"]

        # Convert target points to ego vehicle's local coordinate frame
        ego_target_point = t_u.inverse_conversion_2d(target_point_2d, ego_position, ego_orientation).tolist()
        ego_next_target_point = t_u.inverse_conversion_2d(next_target_point_2d, ego_position, ego_orientation).tolist()
        ego_target_point_world_cord = target_point_2d
        ego_next_target_point_world_cord = next_target_point_2d
        ego_aim_point = t_u.inverse_conversion_2d(self.aim_wp[:2], ego_position, ego_orientation).tolist()

        # Get the remaining route points in the local coordinate frame
        dense_route = []
        dense_route_original = []
        remaining_route = self.remaining_route[:self.config.num_route_points_saved]
        remaining_route_original = self.remaining_route_original[:self.config.num_route_points_saved]

        changed_route = bool(
            (self._waypoint_planner.route_points[self._waypoint_planner.route_index]
             != self._waypoint_planner.original_route_points[self._waypoint_planner.route_index]).any())
        for (checkpoint, checkpoint_original) in zip(remaining_route, remaining_route_original):
            dense_route.append(t_u.inverse_conversion_2d(checkpoint[:2], ego_position[:2], ego_orientation).tolist())
            dense_route_original.append(
                t_u.inverse_conversion_2d(checkpoint_original[:2], ego_position[:2], ego_orientation).tolist())

        # Extract speed reduction object information
        # speed_reduced_by_obj_type, speed_reduced_by_obj_id, speed_reduced_by_obj_distance = None, None, None
        # if speed_reduced_by_obj is not None:
        #     speed_reduced_by_obj_type, speed_reduced_by_obj_id, speed_reduced_by_obj_distance = speed_reduced_by_obj[1:]

        data = {
            "pos_global": ego_position.tolist(),
            "theta": ego_orientation,
            "speed": ego_speed,
            "target_speed": target_speed,
            "speed_limit": speed_limit,
            "target_point": ego_target_point,
            "target_point_next": ego_next_target_point,
            "target_point_world_cord": list(ego_target_point_world_cord),
            "target_point_next_world_cord": list(ego_next_target_point_world_cord),
            "command": self.commands[-2],
            "next_command": self.next_commands[-2],
            "aim_wp": ego_aim_point,
            "route": dense_route,
            "route_original": dense_route_original,
            # "changed_route": changed_route,
            # "speed_reduced_by_obj_type": speed_reduced_by_obj_type,
            # "speed_reduced_by_obj_id": speed_reduced_by_obj_id,
            # "speed_reduced_by_obj_distance": speed_reduced_by_obj_distance,
            "steer": steering,
            "throttle": throttle,
            "brake": bool(brake),
            "control_brake": bool(control_brake),
            "junction": bool(self.junction),
            # "vehicle_hazard": bool(self.vehicle_hazard),
            # "vehicle_affecting_id": self.vehicle_affecting_id,
            # "light_hazard": bool(self.traffic_light_hazard),
            # "walker_hazard": bool(self.walker_hazard),
            # "walker_affecting_id": self.walker_affecting_id,
            # "stop_sign_hazard": bool(self.stop_sign_hazard),
            # "stop_sign_close": bool(self.stop_sign_close),
            # "walker_close": bool(self.walker_close),
            # "walker_close_id": self.walker_close_id,
            "angle": self.angle,
            "augmentation_translation": self.augmentation_translation,
            "augmentation_rotation": self.augmentation_rotation,
            "ego_matrix": self._vehicle.get_transform().get_matrix()
        }

        if self.tp_stats:
            deg_pred_angle = -math.degrees(math.atan2(-ego_aim_point[1], ego_aim_point[0]))

            tp_angle = -math.degrees(math.atan2(-ego_target_point[1], ego_target_point[0]))
            if abs(tp_angle) > 1.0 and abs(deg_pred_angle) > 1.0:
                same_direction = float(tp_angle * deg_pred_angle >= 0.0)
                self.tp_sign_agrees_with_angle.append(same_direction)

        self.last_measurement_data = data

        if ((self.step % self.config.data_save_freq == 0) and (self.save_path is not None) and self.datagen):
            measurements_file = self.save_path / "measurements" / f"{frame:05}.json.gz"
            with gzip.open(measurements_file, "wt", encoding="utf-8") as f:
                ujson.dump(data, f, indent=4)

        return data

    def destroy(self, results=None):
        """
        Save the collected data and statistics to files, and clean up the data structures.
        This method should be called at the end of the data collection process.

        Args:
            results (optional): Any additional results to be processed or saved.
        """
        if self.save_path is not None:
            self.lon_logger.dump_to_json()

            # Save the target speed histogram to a compressed JSON file
            if len(self.speed_histogram) > 0:
                with gzip.open(self.save_path / "target_speeds.json.gz", "wt", encoding="utf-8") as f:
                    ujson.dump(self.speed_histogram, f, indent=4)

            del self.speed_histogram

            if self.tp_stats:
                if len(self.tp_sign_agrees_with_angle) > 0:
                    print("Agreement between TP and steering: ",
                          sum(self.tp_sign_agrees_with_angle) / len(self.tp_sign_agrees_with_angle))
                    with gzip.open(self.save_path / "tp_agreements.json.gz", "wt", encoding="utf-8") as f:
                        ujson.dump(self.tp_sign_agrees_with_angle, f, indent=4)

        del self.tp_sign_agrees_with_angle
        del self.visible_walker_ids
        del self.walker_past_pos

    def _get_steer(self, route_points, current_position, current_heading, current_speed):
        """
        Calculate the steering angle based on the current position, heading, speed, and the route points.

        Args:
            route_points (numpy.ndarray): An array of (x, y) coordinates representing the route points.
            current_position (tuple): The current position (x, y) of the vehicle.
            current_heading (float): The current heading angle (in radians) of the vehicle.
            current_speed (float): The current speed of the vehicle (in m/s).

        Returns:
            float: The calculated steering angle.
        """
        speed_scale = self.config.lateral_pid_speed_scale
        speed_offset = self.config.lateral_pid_speed_offset

        # Calculate the lookahead index based on the current speed
        speed_in_kmph = current_speed * 3.6
        lookahead_distance = speed_scale * speed_in_kmph + speed_offset
        lookahead_distance = np.clip(lookahead_distance, self.config.lateral_pid_default_lookahead,
                                     self.config.lateral_pid_maximum_lookahead_distance)
        lookahead_index = int(min(lookahead_distance, route_points.shape[0] - 1))

        # Get the target point from the route points
        target_point = route_points[lookahead_index]

        # Calculate the angle between the current heading and the target point
        angle_unnorm = self._get_angle_to(current_position, current_heading, target_point)
        normalized_angle = angle_unnorm / 90

        self.aim_wp = target_point
        self.angle = normalized_angle

        # Calculate the steering angle using the turn controller
        steering_angle = self._turn_controller.step(route_points, current_speed, current_position, current_heading)
        steering_angle = round(steering_angle, 3)

        return steering_angle

    def _get_angle_to(self, current_position, current_heading, target_position):
        """
        Calculate the angle (in degrees) from the current position and heading to a target position.

        Args:
            current_position (list): A list of (x, y) coordinates representing the current position.
            current_heading (float): The current heading angle in radians.
            target_position (tuple or list): A tuple or list of (x, y) coordinates representing the target position.

        Returns:
            float: The angle (in degrees) from the current position and heading to the target position.
        """
        cos_heading = math.cos(current_heading)
        sin_heading = math.sin(current_heading)

        # Calculate the vector from the current position to the target position
        position_delta = target_position - current_position

        # Calculate the dot product of the position delta vector and the current heading vector
        aim_x = cos_heading * position_delta[0] + sin_heading * position_delta[1]
        aim_y = -sin_heading * position_delta[0] + cos_heading * position_delta[1]

        # Calculate the angle (in radians) from the current heading to the target position
        angle_radians = -math.atan2(-aim_y, aim_x)

        # Convert the angle from radians to degrees
        angle_degrees = np.float_(math.degrees(angle_radians))

        return angle_degrees
