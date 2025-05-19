from .offline_map_calculations import *
from .graph_utils import *
from .hyper_params import *
from io_utils import print_debug

def generate_ego_vehicle_actions(self, ego_vehicle_data, pedestrians, ego_data, important_objects, key_object_infos, 
                                    vehicles_by_id, traffic_light_info, traffic_sign_info, static_objects, 
                                    measurements, scene_data, scenario_name, stop_signs, traffic_sign_object_tags, 
                                    traffic_light_object_tags):
    """
    Answers the questions:
    Does the ego vehicle need to brake? Why?
    What should the ego vehicle do based on the {actor_name}?
    What is the current speed limit?

    Args:
        ego_vehicle_data (dict): A dictionary containing information about the ego vehicle.
        ego_data (dict): A dictionary containing additional information about the ego vehicle.
        important_objects (list): A list of important objects in the scene.
        key_object_infos (dict): A dictionary containing information about key objects in the scene.
        vehicles_by_id (dict): A dictionary mapping vehicle IDs to vehicle information.
        traffic_light_info (dict): A dictionary containing information about the traffic light.
        stop_sign_info (dict): A dictionary containing information about the stop sign.
        static_objects (list): A list of static objects in the scene.
        landmarks (list): A list of landmarks in the scene.
        measurements (dict): A dictionary containing sensor measurements.
        scenario_name (str): The name of the current scenario.

    Returns:
        tuple: A tuple containing:
            - ego_actions (list): A list of dictionaries representing actions the ego vehicle should take.
            - important_objects (list): The updated list of important objects in the scene.
            - key_object_infos (dict): The updated dictionary containing information about key objects in the scene.
    """

    def del_object_in_key_info(key_dict, obj_list):
        # avoid repeat
        for obj in obj_list:
            obj_id = obj['id']
            keys_to_remove = [k for k, v in key_dict.items() if v.get('id') == obj_id]
            for key in keys_to_remove:
                del key_dict[key]
    
    def is_object_in_key_object(key_object_infos, obj_dict):
        is_found_in_key = False
        for key, key_dict in key_object_infos.items():
            if key_dict['id'] == obj_dict['id']:
                is_found_in_key = True
        return is_found_in_key
    
    def flip_true_in_key_object(key_object_infos, obj_dict, attribute):
        for key, key_dict in key_object_infos.items():
            if key_dict['id'] == obj_dict['id']:
                key_dict[attribute] = True
    
    def consider_vehicle(obj_list):
        for obj in obj_list:
            for key, value in self.all_vehicles_info.items():
                if value['obj_id'] == obj['id']:
                    value['consider'] = True

    def is_vehicle_considered(vehicle):
        for key, value in self.all_vehicles_info.items():
            if value['obj_id'] == vehicle['id']:
                return value.get('consider', False)
    
    def modify_vehicle_info(obj_id, new_key, new_value):
        for key, value in self.all_vehicles_info.items():
            if value['obj_id'] == obj_id:
                value[new_key] = new_value
    
    def set_vehicle_overlap(obj_id, cross_reason, cross_action):
        modify_vehicle_info(obj_id, 'cross_flag', True)
        modify_vehicle_info(obj_id, 'cross_reason', cross_reason)
        modify_vehicle_info(obj_id, 'cross_action', cross_action)

    def add_speed_limit_question(qas_conversation_ego, measurements):        
        """
        Answers "What is the current speed limit?".

        Args:
            qas_conversation_ego (list): A list of dictionaries representing the conversation.
            measurements (dict): A dictionary containing sensor measurements.

        Returns:
            None
        """
        question = "What is the current speed limit?"

        speed_limit = int(measurements['speed_limit'])
        if speed_limit >= 999:
            answer = f"There's no speed limit now"
            if self.future_speed_limit < 999:
                answer = f"{answer}, but soon the speed limit will be {self.future_speed_limit} km/h " + \
                            "because of the speed limit sign ahead."
            else:
                answer = f"{answer}."
        else:
            answer = f"The current speed limit is {speed_limit} km/h"
            if self.future_speed_limit < 999:
                answer = f"{answer}, and soon the speed limit will be {self.future_speed_limit} km/h " + \
                            "because of the speed limit sign ahead."
            else:
                answer = f"{answer}."

        self.add_qas_questions(qa_list=qas_conversation_ego,
                                qid=7, 
                                chain=3,
                                layer=7,
                                qa_type='perception',
                                connection_up=[(6, 0)] ,
                                connection_down=[(3, 2)],
                                question=question,
                                answer=answer)


    def get_rough_position(actor):
        actor_rel_position = transform_to_ego_coordinates(actor['location'], ego_data['world2ego'])
        rough_pos_str = f'{get_pos_str(actor_rel_position)} of it'
        return rough_pos_str

    def get_vehicle_type(vehicle):
        return vehicle.get('base_type') if vehicle.get('base_type') is not None else 'vehicle'
    
    def get_vehicle_color(vehicle):
        color = rgb_to_color_name(vehicle.get("color")) + ' ' if vehicle.get("color") is not None and \
                                            vehicle.get("color") != 'None' else ''
        return color

    def determine_braking_requirement(qas_conversation_ego, pedestrians, measurements, scene_data, vehicles, ego_vehicle, 
                                        scenario_type, traffic_light_info, stop_sign_info, static_objects, 
                                        target_lane_occupied, target_lane_back_clear, lane_change_dir, 
                                        still_changing_lane_flag, highway_merging_danger_flag,
                                        already_stopped_at_stop_sign, must_stop_str, must_brake_str,
                                        obstacle_obj_tag, change_lane_on_highway, extra_changing_reason):
        """
        This function has massive modification because the agent we use is not rule-based.
        Answers "Does the ego vehicle need to brake? Why?".

        Args:
            qas_conversation_ego (list): A list of dictionaries representing the conversation.
            measurements (dict): A dictionary containing sensor measurements.
            vehicles (dict): A dictionary mapping vehicle IDs to vehicle information.
            ego_vehicle (dict): A dictionary containing information about the ego vehicle.
            is_highway (bool): Whether the scenario is on a highway or not.
            scenario_type (str): The type of scenario being evaluated.
            bicycle_scenario (bool): Whether the scenario involves a bicycle or not.
            bicycle_in_junction (bool): Whether the bicycle is in a junction or not.
            blocked_intersection_scenario (bool): Whether the scenario involves a blocked intersection or not.
            traffic_light_info (dict): A dictionary containing information about the traffic light.
            stop_sign_info (dict): A dictionary containing information about the stop sign.

        Returns:
            final_brake_flag (bool)
            final_stop_flag (bool)
        """
        question = "Does the ego vehicle need to brake? Why?"
        answer = "There is no reason for the ego vehicle to brake."

        object_tags = []

        final_brake_flag = False
        final_stop_flag = False

        full_scenario_name = scenario_type
        if self.last_full_scenario_name != full_scenario_name:
            self.leftmost_pos_of_left_hazard = {}
        self.last_full_scenario_name = full_scenario_name
        scenario_type = scenario_type.split('_')[0]
        scenario_name = scenario_type # why is it not unified?

        if self.current_measurement_index - self.last_special_move_index > self.scenario_ignore_interval and \
           not (scenario_name in self.circumvent_scenarios and self.not_passed_circumvent_obstacles):
            scenario_name = 'Normal' # ignore special cases
            scenario_type = 'Normal'

        # print(f'[debug] leftmost_pos = {self.leftmost_pos_of_left_hazard}')

        # acc = get_acceleration_by_future(self.current_measurement_path, 6)
        flags = get_affect_flags(scene_data)

        predict_second = HAZARD_PREDICT_TIME
        predict_frame = int(predict_second * self.frame_rate)
        static_threshold = HAZARD_STATIC_THRESHOLD
        predict_distance = max(ego_vehicle['speed'] * predict_second, static_threshold)

        # print_debug(f"[debug] current frame is {self.current_measurement_index}")
        # print_debug(f"[debug] scenaro_name = {scenario_name}, scenaro_type = {scenario_type}")
        # print_debug(f"[debug] last_special_move = {self.last_special_move_index}")
        # hazardous_walkers = get_walker_hazard_with_prediction(scene_data, prediction_time=15)
        # hazardous_actors = get_all_hazard_with_prediction_sorted(scene_data, prediction_time=15) 
        if not self.in_carla:
            hazardous_walkers = get_hazard_by_future(self.current_measurement_path, self.map, 
                                                        k=predict_frame, filter='walker', max_distance=predict_distance)
            hazardous_actors = get_hazard_by_future(self.current_measurement_path, self.map, 
                                                        k=predict_frame, filter=None, max_distance=predict_distance)
            static_hazardous_actors = get_hazard_by_future(self.current_measurement_path, self.map, 
                                                        k=1, filter=None, max_distance=static_threshold)
        else:
            hazardous_walkers = get_hazard_by_current_frame(self.current_measurement_dict,
                                                            self.map, filter='walker',
                                                            max_distance=predict_distance)
            hazardous_actors = get_hazard_by_current_frame(self.current_measurement_dict,
                                                           self.map, filter=None,
                                                           max_distance=predict_distance)
            static_hazardous_actors = get_hazard_by_current_frame(self.current_measurement_dict,
                                                                  self.map, filter=None,
                                                                  max_distance=static_threshold)

        cuts_in_actor_ids = [actor['id'] for actor in scene_data if actor.get('vehicle_cuts_in', False) == True 
                                                                    and actor['distance'] < CUT_IN_CONSIDER_DISTANCE 
                                                                    and actor['position'][0] > BACK_CONSIDER_THRESHOLD
                                                                    and actor['num_points'] >= MIN_OBJECT_NUM_POINT]
        hazardous_actor_ids = [actor['id'] for actor in hazardous_actors]
        static_hazardous_ids = [actor['id'] for actor in static_hazardous_actors if actor['distance'] < static_threshold]
        too_dangerous_actors = [actor for actor in scene_data if vehicle_is_too_dangerous(actor)]
        print_debug(f"[debug] too_dangerous_actors: {[x['id'] for x in too_dangerous_actors]}")
        hazardous_actors.append(too_dangerous_actors)

        vector_must_stop_actors = []
        vector_must_stop_actors.extend(too_dangerous_actors)
        vector_acc_actors = []
        command_int = get_command_int_by_current_measurement(measurements,
                                                             ego_vehicle)
        
        # when merging, the ego vehicle should check all vehicles which will block the way
        # or it is still far and the ego vehicle must take a bold action to pass it
        # when a chance occured, vehicle in 'acc_list' will be put into a whitelist
        # in which the vehicles will not be considereda danger in few future frames
        # and the ego vehicle would complete the behaviour of passing
        if ego_data['is_in_junction'] and command_int not in [1, 2, 3]:
            if ego_data['steer'] < -TURN_STEER_THRESHOLD: command_int = 1
            elif ego_data['steer'] > TURN_STEER_THRESHOLD: command_int = 2
        
        if scenario_name in self.merging_scenarios and command_int in [1, 2, 3] and \
           ego_data['distance_to_junction'] < JUNCTION_EXTEND_DISTANCE:
            for actor in scene_data:
                if actor['class'] == 'vehicle':
                    res = determine_basic_vector_crossing(command_int, actor,
                                                        whitelist={},
                                                        following_list=self.vehicle_ids_following_ego,
                                                        first_appear_history=self.first_appear_intervals)
                    actor_cross_flag, cross_reason, _, actor_stop_flag, threshold_tuple = res
                    
                    if actor['id'] in self.accelerate_black_list and actor['approaching_dot_product'] < 0 and \
                    not (actor['id'] in self.accelerate_white_list):
                        actor_cross_flag = True
                        actor_stop_flag = True
                    
                    # actor_considered = self.should_consider_vehicle(actor)
                    if actor_cross_flag: # and actor_considered:
                        if actor_stop_flag:
                            print_debug(f"[debug] vector stop actor: id={actor['id']}, type_id={actor['type_id']}, base_type={actor.get('base_type', 'NONE!')}")
                            vector_must_stop_actors.append(actor)
                            if actor['approaching_dot_product'] < 0 and actor['id'] not in self.accelerate_white_list:
                                self.accelerate_black_list[actor['id']] = actor['distance']
                            if command_int in [1, 2, 3]:
                                if actor['id'] not in self.first_appear_intervals:
                                    self.first_appear_intervals[actor['id']] = threshold_tuple
                        else:
                            print_debug(f"[debug] vector acc actor: id={actor['id']}, type_id={actor['type_id']}, base_type={actor.get('base_type', 'NONE!')}")
                            vector_acc_actors.append(actor)

            print_debug(f"[debug] original vector_acc: {[x['id'] for x in vector_acc_actors]}, vector_stop: {[x['id'] for x in vector_must_stop_actors]}")
                
            if command_int in [1, 2]:
                if len(vector_must_stop_actors) <= 0 and len(vector_acc_actors) > 0:
                    for acc_actor in vector_acc_actors:
                        self.accelerate_white_list[acc_actor['id']] = acc_actor['distance']
                        
                
                acc_dist = INF_MAX
                for key in self.accelerate_white_list:
                    if self.accelerate_white_list[key] < acc_dist:
                        acc_dist = self.accelerate_white_list[key]
                
                original_acc_ids = [x['id'] for x in vector_acc_actors]
                for actor in vector_must_stop_actors: 
                    # move misjudged danger vehicle to accelerate list
                    if actor['class'] == 'vehicle':
                        if actor['id'] in self.accelerate_white_list and \
                        actor['id'] not in original_acc_ids:
                            vector_acc_actors.append(actor)
                        if actor['distance'] > acc_dist and \
                        actor['id'] not in original_acc_ids:
                            vector_acc_actors.append(actor)
                            self.accelerate_white_list[actor['id']] = actor['distance']
                
                vector_must_stop_actors = [x for x in vector_must_stop_actors if (not(x['id'] in self.accelerate_white_list)) and \
                                                                                    (actor['distance'] <= acc_dist)]
            
                print_debug(f"[debug] adjusted vector_acc: {[x['id'] for x in vector_acc_actors]}, vector_stop: {[x['id'] for x in vector_must_stop_actors]}")

        combined_actors = [
            actor for actor in scene_data if actor.get('id') in hazardous_actor_ids or \
                actor.get('id') in cuts_in_actor_ids or \
                actor.get('id') in static_hazardous_ids
        ]

        hazardous_actors = sorted(combined_actors, key=lambda actor: actor.get('distance', float('inf')))
        
        if isinstance(stop_sign_info, list):
            stop_sign_info = [x for x in stop_sign_info if x is not None and x['position'][0] > 0.0]
            if len(stop_sign_info) > 0: stop_sign_info = stop_sign_info[0]
            else: stop_sign_info = None
        if isinstance(traffic_sign_info, list):  
            traffic_light_info = [x for x in traffic_light_info if x is not None and x['position'][0] > 0.0]
            if len(traffic_light_info) > 0: traffic_light_info = traffic_light_info[0]
            else: traffic_light_info = None
        
        hazardous = len(hazardous_actors) > 0
        measurements['speed_limit'] = self.current_speed_limit # km/h
        measurements['vehicle_hazard'] = hazardous and 'vehicle' in hazardous_actors[0]['class']
        if (hazardous):
            measurements['speed_reduced_by_obj'] = hazardous_actors[0]
            measurements['speed_reduced_by_obj_id'] = hazardous_actors[0]['id']
            measurements['speed_reduced_by_obj_type'] = hazardous_actors[0]['type_id']
            measurements['speed_reduced_by_obj_distance'] = hazardous_actors[0]['distance']
            ######## for [debug] ########
            self.appended_measurements['speed_reduced_by_obj'] = measurements['speed_reduced_by_obj']
            self.appended_measurements['speed_reduced_by_obj_id'] = measurements['speed_reduced_by_obj_id']
            self.appended_measurements['speed_reduced_by_obj_type'] = measurements['speed_reduced_by_obj_type']
            self.appended_measurements['speed_reduced_by_obj_distance'] = measurements['speed_reduced_by_obj_distance']
            ######## for [debug] ########

        ######## for [debug] ########
        # self.appended_measurements['future_acceleration'] = acc
        self.appended_measurements['affect_flags'] = flags
        self.appended_measurements['hazardous_walkers'] = hazardous_walkers
        self.appended_measurements['hazardous_actors'] = hazardous_actors
        self.appended_measurements['hazardous_flag'] = hazardous
        self.appended_measurements['speed_limit'] = measurements['speed_limit']
        self.appended_measurements['vehicle_hazard_flag'] = measurements['vehicle_hazard']
        ######## for [debug] ########
        
        # speed / speed_limit > 1.031266635497984, done by the controller
        limit_speed = float(measurements['speed_limit']) / 3.6
        junction_speed = MAX_SPEED_IN_JUNCTION

        suggest_stop_flag = False 
        # when obstacle is ahead, the ego vehicle can either continue driving (when obstacle is far) or stop.
        
        if measurements['speed'] / limit_speed > OVER_SPEED_LIMIT_RADIUS:
            answer = "The ego vehicle should brake because it is faster than speed limit."
            final_brake_flag = True
            final_stop_flag = False

        elif ego_vehicle['is_in_junction'] and measurements['speed'] / junction_speed > OVER_SPEED_LIMIT_RADIUS:
            answer = "The ego vehicle should brake because it should slow down in junction."
            final_brake_flag = True
            final_stop_flag = False
            
        # if (measurements['control_brake'] or acc == "Decelerate") or hazardous:
        if True:
            if command_int in [1, 2, 3]:
                command_desc = {
                    3: "go straight",
                    1: "turn left",
                    2: "turn right"
                }
                if len(vector_acc_actors) > 0:
                    closest_actor = min(vector_acc_actors, key=lambda x: x["distance"])
                    object_tags = self.get_key_of_key_object(key_object_infos, object_dict=closest_actor)
                    _, object_desc, _ = get_vehicle_str(closest_actor)
                    final_stop_flag = False
                    final_brake_flag = False
                    answer = f"The ego vehicle should accelerate to {command_desc[command_int]} before {object_desc}({object_tags}) "\
                            "got too close."
                
                # if len(vector_must_stop_actors) > 0:
                #     closest_actor = min(vector_must_stop_actors, key=lambda x: x["distance"])
                #     object_tags = self.get_key_of_key_object(key_object_infos, object_dict=closest_actor)
                #     _, object_desc, _ = get_vehicle_str(closest_actor)
                #     final_stop_flag = True
                #     final_brake_flag = True
                #     answer = f"The ego vehicle must stop because {object_desc}({object_tags}) "\
                #             "is blocking the ego vehicle's path."

            if len(hazardous_walkers) and scenario_name not in \
                ['DynamicObjectCrossing', 'ParkingCrossingPedestrian', 'PedestrianCrossing', 'VehicleTurningRoutePedestrian']:
                closest_pedestrian_idx = np.argmin([x['distance'] for x in pedestrians])
                closest_pedestrian = pedestrians[closest_pedestrian_idx]
                closest_pedestrian_distance = closest_pedestrian['distance']
                brake_or_slow_down = 'stop' if closest_pedestrian_distance < PEDESTRIAN_STOP_DISTANCE else 'slow down'
                final_stop_flag = True if closest_pedestrian_distance < PEDESTRIAN_STOP_DISTANCE else False
                final_brake_flag = True
                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=closest_pedestrian)
                if len(pedestrians) > 1:
                    answer = f"The ego vehicle should {brake_or_slow_down} because of the pedestrians({object_tags}) "\
                                "that are crossing the road."
                else:
                    answer = f"The ego vehicle should {brake_or_slow_down} because of the pedestrian({object_tags}) "\
                                "that is crossing the road."
            
            # beware! below contents are scenario-specified. take special care of them.
            else:
                vehicles_in_front = []
                check_back = False if ego_data['lane_change'] in [0] else True
                # lane_occupied_flag = get_clear_distance_of_lane(vehicles_by_id.values(), -1, check_back) < self.lane_clear_threshold + 20.0
                lane_occupied_flag = target_lane_occupied #  and check_back
                target_lane_back_clear = target_lane_back_clear and check_back
                lane_str = 'nearby lane'
                if lane_change_dir in [1]: lane_str = 'right lane'
                if lane_change_dir in [2, 3]: lane_str = 'left lane'
                if (lane_change_dir in [2] and ego_data['lane_change'] in [0, 1]):
                    lane_str = 'opposite lane'
                lane_change_flag = lane_change_dir in [1, 2, 3]
                print_debug(f"[debug] in determine_braking_requirements, lane_change_dir = {lane_change_dir}, lane_change_flag = {lane_change_flag}")
                
                occupied_str = ', which is occupied,' if lane_occupied_flag else ', which is busy,'

                if not (ego_data['speed'] > STRAIGHT_DRIVE_MIN_SPEED and \
                        abs(measurements['steer']) > STRAIGHT_DRIVE_MIN_STEER): # when driving straight
                    vehicles_in_front = [x for x in vehicles_by_id.values() if self.is_object_in_front(x)]
                    if ego_data['is_in_junction']:
                        vehicles_in_front = [x for x in vehicles_in_front if x['distance'] <= JUNCTION_BRAKE_CONSIDER_RADIUS]
                        # print_debug(f"[debug] current frame = {self.current_measurement_index}")
                        # for vehicle in vehicles_in_front:
                        #     print_debug(f"[debug] {vehicle['id']}")
                    else:
                        vehicles_in_front = [x for x in vehicles_in_front if x['lane_relative_to_ego'] == 0]
                        vehicles_in_front = [x for x in vehicles_in_front if x['distance'] <= NORMAL_BRAKE_CONSIDER_RADIUS \
                                                or x['speed'] < (ego_vehicle['speed'] * SLOW_VEHICLE_RATIO) \
                                                or x['speed'] < SLOW_VEHICLE_SPEED \
                                                or (x.get('light_state', 'None') == 'Brake')
                                                or x.get('brake', 0.0) > VEHICLE_BRAKE_THRESHOLD]
                    
                    vehicles_in_front = sorted(vehicles_in_front, key=lambda actor: actor.get('distance', float('inf')))

                if isinstance(obstacle_obj_tag, list) and len(obstacle_obj_tag) > 0:
                    obstacle_obj_tag = obstacle_obj_tag[0]
                if isinstance(obstacle_obj_tag, str):
                    obs_distance = key_object_infos.get(obstacle_obj_tag, {}).get('distance', INF_MAX)
                    # print(obstacle_obj_tag)
                else:
                    obstacle_obj_tag = None
                    obs_distance = INF_MAX
                
                if change_lane_on_highway and lane_occupied_flag and \
                   (ego_data.get('distance_to_junction', INF_MAX) < HIGHWAY_WAIT_FOR_LANE_CHANGE_DISTANCE or ego_data.get('is_in_junction', False)) and \
                   ego_data['speed'] > ego_data.get('distance_to_junction', INF_MAX) / HIGHWAY_WAIT_FOR_LANE_CHANGE_RATIO:
                    final_brake_flag = True
                    final_stop_flag = False
                    answer = f"The ego vehicle may drive slowly or stop to wait for a chance to change to the {lane_str}."
                
                if self.in_carla and scenario_name == 'BlockedIntersection' and self.role_actor is not None:
                    block_role = None
                    if self.role_actor is not None:
                        for actor in scene_data:
                            if actor['id'] == str(self.role_actor.id):
                                block_role = actor

                    if block_role is not None \
                        and self.should_consider_vehicle(block_role) \
                        and block_role['distance'] < BLOCKED_INTERSECTION_CONSIDER_DISTANCE \
                        and block_role['position'][0] > 0:
                        vehicles_in_front = [block_role] + vehicles_in_front
                        cross_reason = "blocks the intersection"
                        cross_action = "continues to exit the intersection"
                        consider_vehicle([block_role])
                        set_vehicle_overlap(block_role['id'], cross_reason, cross_action)
                
                wait_or_stop_str = "stop and wait" if obs_distance < LANE_CHANGE_STOP_OBSTACLE_DISTANCE else "drive cautiously in current lane and wait"
                if "HazardAtSideLane" in scenario_name:
                    wait_or_stop_str = "drive slowly to match the speed of the bicycle and wait"
                suggest_stop_flag = True if wait_or_stop_str == "stop and wait" else False

                if 'Accident' in scenario_name and lane_change_flag:
                    # if hazardous and 'vehicle.dodge.charger_police_2020' == measurements['speed_reduced_by_obj_type']:
                    #     police_cars = [x for x in hazardous_actors if x['type_id'] == 'vehicle.dodge.charger_police_2020']
                    #     car_id = police_cars[0]['id']
                    #     if car_id not in self.leftmost_pos_of_left_hazard:
                    #         self.leftmost_pos_of_left_hazard[car_id] = self.inf_num
                    #     self.leftmost_pos_of_left_hazard[car_id] = \
                    #         min(self.leftmost_pos_of_left_hazard[car_id], police_cars[0]['position'][1])
                    #     delta = 1.2 if self.leftmost_pos_of_left_hazard[car_id] < 1.2 else 2
                    #     police_cars = [x for x in police_cars if x['position'][1] < self.leftmost_pos_of_left_hazard[car_id] + delta]
                    #     if police_cars and lane_occupied_flag:
                    #         object_tags = self.get_key_of_key_object(key_object_infos, object_dict=police_cars[0])
                    obs_distance = key_object_infos.get(obstacle_obj_tag, None)
                    object_tag_str = f"({object_tags})" if object_tags else ""
                    change_reason = extra_changing_reason if extra_changing_reason is not None else f"bypass the accident{object_tag_str}"
                    if lane_occupied_flag:
                        answer = f"The ego vehicle should {wait_or_stop_str} for a chance because it must invade the {lane_str}{occupied_str}"\
                                f" in order to {change_reason}."
                        final_brake_flag = True
                        final_stop_flag = suggest_stop_flag
                        if target_lane_back_clear:
                            answer = f"The ego vehicle may drive slowly and follow the vehicle on {lane_str}"\
                                f" in order to {change_reason}."
                            final_brake_flag = False
                            final_stop_flag = False
                    elif not self.opposite_flag:
                        answer = f"The ego vehicle doesn't need to brake if it wants to take a chance to change to the {lane_str}"\
                                f" in order to {change_reason}."
                        final_brake_flag = False
                        final_stop_flag = False
                    if still_changing_lane_flag:
                        answer = f"The ego vehicle doesn't need to brake because it is changing to the {lane_str} now."
                        final_brake_flag = False
                        final_stop_flag = False

                elif 'ConstructionObstacle' in scenario_name and lane_change_flag:
                    # if hazardous and 'static.prop.trafficwarning' == measurements['speed_reduced_by_obj_type']:
                    #     traffic_warnings = [x for x in hazardous_actors if x['type_id'] == 'static.prop.trafficwarning']
                    #     if traffic_warnings and lane_occupied_flag:
                    #         wid = traffic_warnings[0]['id']
                    #         if wid not in self.leftmost_pos_of_left_hazard:
                    #             self.leftmost_pos_of_left_hazard[wid] = self.inf_num
                    #         self.leftmost_pos_of_left_hazard[wid] = min(self.leftmost_pos_of_left_hazard[wid], traffic_warnings[0]['position'][1])
                    #         delta = 1.2 if self.leftmost_pos_of_left_hazard[wid] < 1.2 else 2
                    #         traffic_warnings = [x for x in traffic_warnings if x['position'][1] < self.leftmost_pos_of_left_hazard[wid] + delta]
                    #         if traffic_warnings:
                    #             object_tags = self.get_key_of_key_object(key_object_infos, object_dict=traffic_warnings[0])
                    obs_distance = key_object_infos.get(obstacle_obj_tag, None)
                    object_tag_str = f"({object_tags})" if object_tags else ""
                    change_reason = extra_changing_reason if extra_changing_reason is not None else f"bypass the construction area{object_tag_str}"
                    if lane_occupied_flag:
                        answer = f"The ego vehicle should {wait_or_stop_str} for a chance because it must invade the {lane_str}{occupied_str}" \
                                    f" in order to {change_reason}."
                        final_brake_flag = True
                        final_stop_flag = suggest_stop_flag
                        if target_lane_back_clear:
                            answer = f"The ego vehicle may adjust its speed and follow the vehicle on {lane_str}"\
                                f" in order to {change_reason}."
                            final_brake_flag = False
                            final_stop_flag = False
                    elif not self.opposite_flag:
                        answer = f"The ego vehicle doesn't need to brake if it wants to take a chance to change to the {lane_str}"\
                                    f" in order to {change_reason}."
                        final_brake_flag = False
                        final_stop_flag = False
                    if still_changing_lane_flag:
                        answer = f"The ego vehicle doesn't need to brake because it is changing to the {lane_str} now."
                        final_brake_flag = False
                        final_stop_flag = False
                    
                elif 'ParkedObstacle' in scenario_name and lane_change_flag:
                    # if hazardous and measurements['speed_reduced_by_obj_id'] in vehicles:
                    #     vehicle = vehicles[measurements['speed_reduced_by_obj_id']]
                    #     object_tags = self.get_key_of_key_object(key_object_infos, object_dict=vehicle)
                    obs_distance = key_object_infos.get(obstacle_obj_tag, None)
                    object_tag_str = f"({object_tags})" if object_tags else ""
                    change_reason = extra_changing_reason if extra_changing_reason is not None else f"bypass the parked vehicle{object_tag_str}"
                    if lane_occupied_flag:
                        answer = f"The ego vehicle should {wait_or_stop_str} for a chance because it must invade the {lane_str}{occupied_str}" \
                                        f" in order to {change_reason}."
                        final_brake_flag = True
                        final_stop_flag = suggest_stop_flag
                        if target_lane_back_clear:
                            answer = f"The ego vehicle may adjust its speed and follow the vehicle on {lane_str}"\
                                f" in order to {change_reason}."
                            final_brake_flag = False
                            final_stop_flag = False
                    elif not self.opposite_flag:
                        answer = f"The ego vehicle doesn't need to brake if it wants to take a chance to change to the {lane_str}"\
                                    f" in order to {change_reason}."
                        final_brake_flag = False
                        final_stop_flag = False
                    if still_changing_lane_flag:
                        answer = f"The ego vehicle doesn't need to brake because it is changing to the {lane_str} now."
                        final_brake_flag = False
                        final_stop_flag = False
                            
                elif 'VehicleOpensDoor' in scenario_name and lane_change_flag:
                    # vehicles_open_door = [v for v in vehicles_by_id.values() if v['type_id'] == 'vehicle.mercedes.coupe_2020' 
                    #             and v['position'][0] > 0
                    #             and v['position'][1] > 0.5 
                    #             and (float(v['distance']) <= 22.0 and v['speed'] < 0.1) and is_vehicle_in_camera(self.CAMERA_FRONT, v)]
                    # if vehicles_open_door and (len(self.role_vehicle_ids) == 0 or vehicles_open_door[0]['id'] in self.role_vehicle_ids) \
                    #     and lane_occupied_flag:
                    #     vehicle = vehicles_open_door[0]
                    #     self.role_vehicle_ids.append(vehicle['id']) # avoid misjudge
                    #     object_tags = self.get_key_of_key_object(key_object_infos, object_dict=vehicle)
                    obs_distance = key_object_infos.get(obstacle_obj_tag, None)
                    object_tag_str = f"({object_tags})" if object_tags else ""
                    change_reason = extra_changing_reason if extra_changing_reason is not None else f"bypass the vehicle with the opened doors{object_tag_str}"
                    if lane_occupied_flag:
                        answer = f"The ego vehicle should {wait_or_stop_str} for a chance because it must invade the {lane_str}{occupied_str}" \
                                                    f" in order to {change_reason}."
                        final_brake_flag = True
                        final_stop_flag = suggest_stop_flag
                        if target_lane_back_clear:
                            answer = f"The ego vehicle may adjust its speed and follow the vehicle on {lane_str}"\
                                f" in order to {change_reason}."
                            final_brake_flag = False
                            final_stop_flag = False
                    elif not self.opposite_flag:
                        answer = f"The ego vehicle doesn't need to brake if it wants to take a chance to change to the {lane_str}"\
                                    f" in order to {change_reason}."
                        final_brake_flag = False
                        final_stop_flag = False
                    if still_changing_lane_flag:
                        answer = f"The ego vehicle doesn't need to brake because it is changing to the {lane_str} now."
                        final_brake_flag = False
                        final_stop_flag = False

                elif 'HazardAtSideLane' in scenario_name and lane_change_flag:
                    # bicycles = [v for v in vehicles_by_id.values() if v['base_type'] == 'bicycle' 
                    #                                                 and self.should_consider_vehicle(v) 
                    #                                                 and float(v['distance']) < 40
                    #                                                 and v['lane_relative_to_ego'] == 0]
                    # if bicycles:
                    #     bicycles.sort(key=lambda x:x['distance'])
                    #     closest_bicycle = bicycles[0]

                    #     brake_or_stop = 'stop' if measurements['speed'] < 0.5 else 'brake'
                    #     if closest_bicycle['distance'] < 40 and lane_occupied_flag:
                    #         object_tags = self.get_key_of_key_object(key_object_infos, object_dict=closest_bicycle)
                    obs_distance = key_object_infos.get(obstacle_obj_tag, None)
                    object_tag_str = f"({object_tags})" if object_tags else ""
                    change_reason = extra_changing_reason if extra_changing_reason is not None else f"bypass the slow bicycles{object_tag_str}"
                    if lane_occupied_flag:
                        answer = f"The ego vehicle should {wait_or_stop_str} for a chance because it must invade the {lane_str}{occupied_str}" \
                                    f" in order to {change_reason}."
                        final_brake_flag = True
                        final_stop_flag = suggest_stop_flag
                        if target_lane_back_clear:
                            answer = f"The ego vehicle may adjust its speed and follow the vehicle on {lane_str}"\
                                f" in order to {change_reason}."
                            final_brake_flag = False
                            final_stop_flag = False
                    elif not self.opposite_flag:
                        answer = f"The ego vehicle doesn't need to brake if it wants to take a chance to change to the {lane_str}"\
                                    f" in order to {change_reason}."
                        final_brake_flag = False
                        final_stop_flag = False
                    if still_changing_lane_flag:
                        answer = f"The ego vehicle doesn't need to brake because it is changing to the {lane_str} now."
                        final_brake_flag = False
                        final_stop_flag = False

                elif 'ParkingExit' in scenario_name and lane_change_flag:
                    obs_distance = key_object_infos.get(obstacle_obj_tag, None)
                    object_tag_str = f"({object_tags})" if object_tags else ""
                    change_reason = extra_changing_reason if extra_changing_reason is not None else f"exit the parking space"
                    if lane_occupied_flag:
                        answer = f"The ego vehicle should stop and wait for a chance because " +\
                                    f"it wants to {change_reason}, but the {lane_str} is occupied."
                        final_brake_flag = True
                        final_stop_flag = True # you can't do "follow lane" in parking spaces!
                        if target_lane_back_clear:
                            answer = f"The ego vehicle may start up and follow the vehicle on {lane_str}"\
                                f" in order to {change_reason}."
                            final_brake_flag = False
                            final_stop_flag = False
                    elif not self.opposite_flag:
                        answer = f"The ego vehicle doesn't need to brake if it wants to take a chance to exit the parking space."
                        final_brake_flag = False
                        final_stop_flag = False
                    if still_changing_lane_flag:
                        changing_str = "changing lane" if extra_changing_reason is not None else "exiting the parking space"
                        answer = f"The ego vehicle doesn't need to brake because it is {changing_str} now."
                        final_brake_flag = False
                        final_stop_flag = False
                elif 'LaneChange' in scenario_name and lane_change_flag:
                    obs_distance = key_object_infos.get(obstacle_obj_tag, None)
                    object_tag_str = f"({object_tags})" if object_tags else ""
                    if lane_occupied_flag:
                        answer = f"The ego vehicle should keep driving slowly in the current lane and wait for a chance because " +\
                                    f"it wants to change lane soon, but the {lane_str} is occupied."
                        final_brake_flag = True
                        final_stop_flag = False
                        if target_lane_back_clear:
                            answer = f"The ego vehicle may follow the vehicle on {lane_str}"\
                                f" in order to change lane."
                            final_brake_flag = False
                            final_stop_flag = False
                    elif not self.opposite_flag:
                        answer = f"The ego vehicle doesn't need to brake if it wants to take a chance to change to the {lane_str}."
                        final_brake_flag = False
                        final_stop_flag = False
                    if still_changing_lane_flag:
                        answer = f"The ego vehicle doesn't need to brake because it is changing lane now."
                        final_brake_flag = False
                        final_stop_flag = False
                elif 'YieldToEmergencyVehicle' in scenario_name and lane_change_flag:
                    print_debug(f"[debug] yield braking branch")
                    obs_distance = key_object_infos.get(obstacle_obj_tag, None)
                    object_tag_str = f"({object_tags})" if object_tags else ""
                    change_reason = extra_changing_reason if extra_changing_reason is not None else f"yield the emergency vehicle from behind"
                    final_brake_flag = False
                    final_stop_flag = False
                    if lane_occupied_flag:
                        print_debug(f"[debug] yield lane occupied branch")
                        answer = f"The ego vehicle should keep driving in the current lane slowly and wait for a chance because " +\
                                 f"it wants to {change_reason}, but the {lane_str} is occupied."
                        final_brake_flag = True
                        final_stop_flag = False
                        if self.in_carla:
                            print_debug(f"[debug] yield lane occupied branch, carla")
                            answer = f"The ego vehicle should keep driving in the current lane slowly and wait for a chance because " +\
                                     f"it wants to {change_reason}, but the {lane_str} is occupied."
                            final_brake_flag = True
                            final_stop_flag = False
                        if target_lane_back_clear:
                            print_debug(f"[debug] yield follow branch")
                            answer = f"The ego vehicle may follow the vehicle on {lane_str}"\
                                f" in order to change lane to {change_reason}."
                            final_brake_flag = False
                            final_stop_flag = False
                    elif not self.opposite_flag:
                        print_debug(f"[debug] yield change branch")
                        answer = f"The ego vehicle doesn't need to brake if it wants to take a chance to change lane to {change_reason}."
                    if still_changing_lane_flag:
                        print_debug(f"[debug] yield changing branch")
                        answer = f"The ego vehicle doesn't need to brake because it is changing lane now."
                            
                elif 'DynamicObjectCrossing' in scenario_name or 'ParkingCrossingPedestrian' in scenario_name or \
                    'PedestrianCrossing' in scenario_name or 'VehicleTurningRoutePedestrian' in scenario_name:
                    # in definition, DynamicObjectCrossing's obstacle might be a bicycle
                    # but it does not exist in b2d
                    peds = [x for x in pedestrians if x['num_points'] >= MIN_PEDESTRIAN_NUM_POINT and
                            0.0 < x['position'][0] < PEDESTRIAN_CONSIDER_DISTANCE
                            and (abs(x['speed']) > PEDESTRIAN_MIN_SPEED or x['distance'] < PEDESTRIAN_STOP_DISTANCE)
                            and (PEDESTRIAN_MIN_Y < x['position'][1] < PEDESTRIAN_MAX_Y or is_vehicle_in_camera(self.CAMERA_FRONT, x))] # carla lane width is 3.5m
                    if peds:
                        closest_pedestrian_idx = np.argmin([x['distance'] for x in peds])
                        closest_pedestrian = peds[closest_pedestrian_idx]
                        closest_pedestrian_distance = closest_pedestrian['distance']
                        brake_or_slow_down = 'stop' if closest_pedestrian_distance < PEDESTRIAN_STOP_DISTANCE else 'slow down'
                        final_brake_flag = True
                        final_stop_flag = True if closest_pedestrian_distance < PEDESTRIAN_STOP_DISTANCE and abs(closest_pedestrian['speed']) > PEDESTRIAN_MIN_SPEED else False
                        for ped in peds:
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=ped)
                        if len(peds) > 1:
                            answer = f"The ego vehicle should {brake_or_slow_down} because of the pedestrians({object_tags}) "\
                                        "that are crossing the road."
                        else:
                            answer = f"The ego vehicle should {brake_or_slow_down} because of the pedestrian({object_tags}) "\
                                        "that is crossing the road."

                elif vehicles_in_front:
                    brake_due_to_leading_vehicle = not measurements['vehicle_hazard']
                    is_highway = False

                    # List of highway scenarios
                    highway_scenarios = [
                        "EnterActorFlow", 
                        "EnterActorFlowV2", 
                        "HighwayCutIn", 
                        "HighwayExit", 
                        "MergerIntoSlowTraffic",
                        "MergerIntoSlowTrafficV2",
                        "YieldToEmergencyVehicle",
                    ]

                    speed_limit = int(measurements['speed_limit'])

                    if scenario_name in highway_scenarios and speed_limit > HIGHWAY_MIN_SPEED:
                        is_highway = True

                    bike_scenario = False
                    blocked_intersection_scenario = False
                    for vehicle in vehicles_in_front:
                        # find bicycles that are of type scenario
                        if 'bicycle' in vehicle['class'] and \
                                        (ego_data['distance_to_junction'] < JUNCTION_EXTEND_DISTANCE or ego_data['is_in_junction']) and \
                                        scenario_name == 'CrossingBicycleFlow':
                            bike_scenario = True
                            color = rgb_to_color_name(vehicle["color"]) + ' ' if vehicle["color"] is not None and \
                                                                            vehicle["color"] != 'None' else ''
                            vehicletype = vehicle['base_type']
                            vehicle_rel_position = transform_to_ego_coordinates(vehicle['location'], ego_data['world2ego'])
                            
                            rough_pos_str = f'{get_pos_str(vehicle_rel_position)} of it'
                            
                        elif vehicle['distance'] < BLOCKED_INTERSECTION_CONSIDER_DISTANCE and scenario_name == 'BlockedIntersection' and self.crossed_junction_flag:
                            blocked_intersection_scenario = True
                    
                    actor_hazard = vehicles_in_front[0]
                    
                    color = get_vehicle_color(actor_hazard)
                    vehicletype = get_vehicle_type(actor_hazard)
                    rough_pos_str = get_rough_position(actor_hazard)

                    considered_vehicle = self.should_consider_vehicle(actor_hazard)
                    if vehicle_is_too_dangerous(actor_hazard):
                        brake_stop_str = "stop"
                        suggest_stop_flag = True
                        hazard_desc = "is too close"
                    elif actor_hazard['speed'] < STOP_VEHICLE_SPEED:
                        if actor_hazard['distance'] < STOP_FOR_STOPPED_VEHICLE_DISTANCE and \
                           not ('HighwayCutIn' in scenario_name and actor_hazard['id'] in self.front_merge_vehicle_ids):
                            brake_stop_str = "stop"
                            suggest_stop_flag = True
                        else:
                            brake_stop_str = "slow down and stop"
                            suggest_stop_flag = False
                        hazard_desc = "is not moving"
                    else:
                        brake_stop_str = "brake"
                        suggest_stop_flag = False
                        hazard_desc = "is driving slowly"

                    # Determine if there is no reason for the ego vehicle to brake
                    if actor_hazard['num_points'] < 3 or not considered_vehicle and \
                       final_stop_flag == False and \
                       final_brake_flag == False:
                        final_brake_flag = False
                        final_stop_flag = False
                        answer = "There is no reason for the ego vehicle to brake."
                    # Handle the case where the hazard vehicle is a leading vehicle
                    elif brake_due_to_leading_vehicle and actor_hazard['distance'] < BRAKE_FOR_LEADING_VEHICLE_DISTANCE:
                        # print_debug(f"[debug] brake = {actor_hazard.get('brake', 0.0)}, light = {actor_hazard.get('light_state', 'None')}, frame = {self.current_measurement_index}")
                        # if actor_hazard['speed'] < 0.5:
                        #     object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                        #     answer = "The ego vehicle should slow down and stop because the " + f"{color}{vehicletype} " +\
                        #                                                                 f"that is {rough_pos_str}({object_tags}) has stopped."
                        # else:
                        if actor_hazard['base_type'] == 'bicycle':
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                            if is_vehicle_pointing_towards_ego(actor_hazard['position'], actor_hazard['yaw'], BICYCLE_CROSS_ANGLE)[0]:
                                final_brake_flag = True
                                final_stop_flag = True
                                answer = "The ego vehicle should stop to wait for the " +\
                                        f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) to pass."
                            elif final_stop_flag == False:
                                final_brake_flag = True
                                final_stop_flag = False
                                answer = "The ego vehicle should slow down because of the " +\
                                        f"{color}{vehicletype} that is {rough_pos_str}({object_tags})."
                        elif is_vehicle_pointing_towards_ego(actor_hazard['position'], actor_hazard['yaw'], VEHICLE_TOWARDS_ANGLE)[0]:
                            if actor_hazard.get('speed', 0.0) > SLOW_VEHICLE_SPEED and \
                               not ego_data.get('is_in_junction', False) and \
                               final_stop_flag == False:
                                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                                final_brake_flag = True
                                final_stop_flag = False
                                answer = "The ego vehicle should brake because the " +\
                                        f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) is driving towards the ego vehicle."
                        elif actor_hazard.get('light_state', "None") == 'Brake' and final_stop_flag == False:
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                            final_brake_flag = True
                            final_stop_flag = False
                            answer = "The ego vehicle should brake because the " +\
                                    f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) braked, which can be seen from the brake lights."
                        elif actor_hazard.get('brake', 0.0) > VEHICLE_BRAKE_THRESHOLD and final_stop_flag == False:
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                            final_brake_flag = True
                            final_stop_flag = False
                            answer = "The ego vehicle should brake because the " +\
                                    f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) braked."
                        elif STOP_VEHICLE_SPEED < actor_hazard.get('speed', SLOW_VEHICLE_SPEED + 1.0) < max(SLOW_VEHICLE_SPEED, ego_vehicle['speed'] * SLOW_VEHICLE_RATIO) and \
                             final_stop_flag == False:
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                            final_brake_flag = True
                            final_stop_flag = False
                            answer = "The ego vehicle should adjust its speed to the speed of the " +\
                                                                f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) because it is driving slowly."
                        elif actor_hazard.get('speed', 5.0) <= STOP_VEHICLE_SPEED:
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                            if (actor_hazard.get('distance', 5.0) > STOP_FOR_STOPPED_VEHICLE_DISTANCE and final_stop_flag == False) or \
                               ('HighwayCutIn' in scenario_name and actor_hazard['id'] in self.front_merge_vehicle_ids):
                                final_brake_flag = True
                                final_stop_flag = False
                                answer = "The ego vehicle should slow down since the " +\
                                                                    f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) is not moving."
                            else:
                                final_brake_flag = True
                                final_stop_flag = True
                                answer = "The ego vehicle should slow down and stop since the " +\
                                                                    f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) is not moving."
                        elif actor_hazard['distance'] < TOO_CLOSE_THRESHOLD and final_stop_flag == False:
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                            final_brake_flag = True
                            final_stop_flag = False
                            answer = f"The ego vehicle should control its speed because the {color}" \
                                                                    f"{vehicletype} that is {rough_pos_str}({object_tags}) is too close."
                    
                    # Handle the case where the scenario is on a highway
                    elif is_highway:
                        object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                        final_brake_flag = True
                        final_stop_flag = suggest_stop_flag
                        answer = f"The ego vehicle should {brake_stop_str} because of the " +\
                                                                f"{color}{vehicletype} that is {rough_pos_str}({object_tags})."
                    # Handle the case where the scenario is not on a highway
                    else:
                        # Check if the ego vehicle is in a junction or near a junction, and the hazard vehicle is
                        # on a different road
                        if (ego_vehicle['is_in_junction'] or (ego_vehicle['distance_to_junction'] is not None and \
                                                                ego_vehicle['distance_to_junction'] < JUNCTION_EXTEND_DISTANCE)) and \
                                                                actor_hazard['road_id'] != ego_vehicle['road_id']:
                            actor_rel_position = transform_to_ego_coordinates(actor_hazard['location'], ego_data['world2ego'])
                            # Determine the direction of the hazard vehicle relative to the junction
                            if actor_rel_position[1] < -JUNCTION_POS_OFFSET:
                                direction_junction = "on the left side of the junction"
                            # right
                            elif actor_rel_position[1] > JUNCTION_POS_OFFSET:
                                direction_junction = "on the right side of the junction"
                            elif actor_hazard['road_id'] != ego_vehicle['road_id']:
                                direction_junction = "on the opposite side of the junction"
                            else:
                                direction_junction = "near the junction"
                                # raise ValueError(f"Unknown position of vehicle {vehicle['id']}.")
                                raise ValueError(f"Unknown position of vehicle.")
                            
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                            final_brake_flag = True
                            final_stop_flag = suggest_stop_flag
                            answer = f"The ego vehicle should {brake_stop_str} because of the {color}" \
                                                                    f"{vehicletype} that is {direction_junction}({object_tags})."
                        # Handle other cases
                        else:
                            if actor_hazard['vehicle_cuts_in']:
                                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                                final_brake_flag = True
                                final_stop_flag = False
                                brake_or_stop_str = "brake"
                                if actor_hazard['distance'] < CUT_IN_STOP_DISTANCE:
                                    final_stop_flag = True
                                    brake_or_stop_str = "stop"
                                answer = f"The ego vehicle should {brake_or_stop_str} because of the {color}"\
                                                    f"{vehicletype}({object_tags}) that is cutting into the ego vehicle's lane."
                            elif vehicle_is_too_dangerous(actor_hazard):
                                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                                final_brake_flag = True
                                final_stop_flag = True
                                answer = f"The ego vehicle should stop immediately because the {color}" \
                                                                        f"{vehicletype} that is {rough_pos_str}({object_tags}) is too close."
                            elif actor_hazard['speed'] < max(SLOW_VEHICLE_SPEED, ego_vehicle['speed'] * SLOW_VEHICLE_RATIO):
                                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                                final_brake_flag = True
                                final_stop_flag = suggest_stop_flag
                                answer = f"The ego vehicle should {brake_stop_str} because the {color}" \
                                                                        f"{vehicletype} that is {rough_pos_str}({object_tags}) {hazard_desc}."
                            elif actor_hazard['distance'] < TOO_CLOSE_THRESHOLD:
                                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                                final_brake_flag = True
                                final_stop_flag = suggest_stop_flag
                                answer = f"The ego vehicle should control its speed because the {color}" \
                                                                        f"{vehicletype} that is {rough_pos_str}({object_tags}) is too close."

                    # Special cases for specific scenarios
                    if scenario_type == 'BlockedIntersection' and (self.in_carla or blocked_intersection_scenario):
                        object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                        if not self.in_carla or (self.role_actor is not None and actor_hazard['id'] == str(self.role_actor.id)):
                            suggest_stop_flag = True if actor_hazard['distance'] < BLOCKED_INTERSECTION_STOP_DISTANCE else False
                            final_brake_flag = True
                            final_stop_flag = suggest_stop_flag
                            suggest_str = "stop" if final_stop_flag else "slow down and stop"
                            answer = f"The ego vehicle should {suggest_str} because the {color}{vehicletype} that is " +\
                                                                    f"{rough_pos_str} is blocking the intersection."

                    if hazardous:
                        if scenario_type == 'CrossingBicycleFlow' and bike_scenario:
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                            final_brake_flag = True
                            final_stop_flag = False
                            answer = f"The ego vehicle should slow down because of the {color}{vehicletype}({object_tags}) " +\
                                                        f"that is {rough_pos_str} and is crossing the intersection."
                        
                        if scenario_type == "InterurbanActorFlow" and ego_vehicle['is_in_junction']:
                            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=actor_hazard)
                            suggest_stop_flag = True if actor_hazard['distance'] < INTERURBAN_ACTOR_FLOW_STOP_DISTANCE else False
                            brake_stop_str = "stop" if actor_hazard['distance'] < INTERURBAN_ACTOR_FLOW_STOP_DISTANCE else "brake"
                            final_brake_flag = True
                            final_stop_flag = suggest_stop_flag
                            answer = f"The ego vehicle should {brake_stop_str} because of the {color}{vehicletype}({object_tags}) that " +\
                                            f"is on the oncoming lane and is crossing paths with the ego vehicle."

        if answer == "There is no reason for the ego vehicle to brake." and ego_vehicle['hazard_detected_40']:
            leading_vehicle_id = ego_vehicle['affects_ego_40']

            if leading_vehicle_id is not None:
                leading_vehicle = vehicles[leading_vehicle_id]
                considered_vehicle = self.should_consider_vehicle(leading_vehicle)
                if considered_vehicle:
                    color = get_vehicle_color(leading_vehicle)
                    vehicletype = get_vehicle_type(leading_vehicle)
                    rough_pos_str = get_rough_position(leading_vehicle)
                    # if measurements['speed'] < (1 / 3.6) * 0.9 * measurements['speed_limit'] \
                    #                                     and measurements['throttle'] < 0.9:
                    if True:
                        object_tags = self.get_key_of_key_object(key_object_infos, object_dict=leading_vehicle)
                        if leading_vehicle['base_type'] == 'bicycle':
                            answer = "The ego vehicle should slow down because of the " +\
                                                                f"{color}{vehicletype} that is {rough_pos_str}({object_tags})."
                            final_brake_flag = True
                            final_stop_flag = False
                        elif STOP_VEHICLE_SPEED < leading_vehicle.get('speed', SLOW_VEHICLE_SPEED + 20.0) < max(SLOW_VEHICLE_SPEED, ego_vehicle['speed'] * SLOW_VEHICLE_RATIO):       
                            answer = "The ego vehicle should slow down to adjust its speed to the speed of the " +\
                                                                f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) because it is driving slowly."
                            final_brake_flag = True
                            final_stop_flag = False
                        elif leading_vehicle.get('speed', SLOW_VEHICLE_SPEED + 20.0) <= STOP_VEHICLE_SPEED:
                            if leading_vehicle.get('distance', 20.0) > STOP_FOR_STOPPED_VEHICLE_DISTANCE or \
                               ('HighwayCutIn' in scenario_name and actor_hazard['id'] in self.front_merge_vehicle_ids):
                                answer = "The ego vehicle should slow down because the " +\
                                                                    f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) is not moving."
                                final_brake_flag = True
                                final_stop_flag = False
                            else:
                                answer = "The ego vehicle should stop because the " +\
                                                                    f"{color}{vehicletype} that is {rough_pos_str}({object_tags}) is not moving."
                                final_brake_flag = True
                                final_stop_flag = True
                
                # Special cases for specific scenarios
                if leading_vehicle['distance'] < BLOCKED_INTERSECTION_CONSIDER_DISTANCE and scenario_name == 'BlockedIntersection':
                    object_tags = self.get_key_of_key_object(key_object_infos, object_dict=leading_vehicle)
                    if not self.in_carla or (self.role_actor is not None and leading_vehicle['id'] == str(self.role_actor.id)):
                        final_brake_flag = True
                        final_stop_flag = True
                        answer = f"The ego vehicle should stop because of the {color}{vehicletype}({object_tags}) that is " +\
                                                                    f"{rough_pos_str} and is blocking the intersection."
        
        if cuts_in_actor_ids and len(cuts_in_actor_ids) > 0:
            cut_in_vehicle = vehicles_by_id[cuts_in_actor_ids[0]]
            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=cut_in_vehicle)
            color = get_vehicle_color(cut_in_vehicle)
            vehicletype = get_vehicle_type(cut_in_vehicle)
            rough_pos_str = get_rough_position(cut_in_vehicle)
            final_brake_flag = True
            final_stop_flag = False
            brake_or_stop_str = "brake"
            if cut_in_vehicle['distance'] < CUT_IN_STOP_DISTANCE:
                final_stop_flag = True
                brake_or_stop_str = "stop"
            if cut_in_vehicle['speed'] >= max(ego_vehicle['speed'] * SLOW_VEHICLE_RATIO, SLOW_VEHICLE_SPEED) and \
               cut_in_vehicle['distance'] > CUT_IN_STOP_DISTANCE:
                final_stop_flag = False
                brake_or_stop_str = "control its speed"
            answer = f"The ego vehicle should {brake_or_stop_str} because of the {color}{vehicletype}({object_tags}) that " +\
                     f"is cutting into the ego vehicle's lane."
        
        if len(vector_must_stop_actors) > 0:
            closest_actor = min(vector_must_stop_actors, key=lambda x: x["distance"])
            object_tags = self.get_key_of_key_object(key_object_infos, object_dict=closest_actor)
            _, object_desc, _ = get_vehicle_str(closest_actor)
            exit_ego_cord = transform_to_ego_coordinates([measurements['junction_exit_wp_x'],
                                                          measurements['junction_exit_wp_y'],
                                                          ego_vehicle['location'][2]],
                                                          ego_vehicle['world2ego'])
            if final_brake_flag == False and \
               ((math.pi * (180 - INTERSECTION_GET_CLOSE_DEGREE) / 180) < closest_actor['yaw'] < (math.pi * (180 + INTERSECTION_GET_CLOSE_DEGREE) / 180) or \
               (math.pi * (-180 - INTERSECTION_GET_CLOSE_DEGREE) / 180) < closest_actor['yaw'] < (math.pi * (-180 + INTERSECTION_GET_CLOSE_DEGREE) / 180)) and \
               abs(closest_actor['position'][1]) > INTERSECTION_GET_CLOSE_Y and \
               not (ego_vehicle['is_in_junction'] and exit_ego_cord[0] <= EXIT_MIN_X):
                final_brake_flag = True
                answer = f"The ego vehicle should move slowly and wait for a chance because the {object_desc}({object_tags}) "\
                        "is blocking the ego vehicle's path."
                # DONT MODIFY 'move slowly and wait for a chance' because it is used for identification in behaviour.py
            else:
                final_stop_flag = True
                final_brake_flag = True
                answer = f"The ego vehicle must stop because the {object_desc}({object_tags}) "\
                        "is blocking the ego vehicle's path."
        
        signal_answer = ""
        previous_brake_flag = final_brake_flag
        previous_stop_flag = final_stop_flag
        if stop_sign_info is not None:
            if flags['affected_by_stop_sign'] == True:
                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=stop_sign_info)
                if already_stopped_at_stop_sign or self.stopped_at_stop_sign:
                    # final_brake_flag = False
                    # final_stop_flag = False
                    signal_answer = "The ego vehicle needn't stop at the stop sign now because it has already stopped at it."
                elif len(object_tags) > 0:
                    distance_to_trigger_location = calculate_2D_distance(location1=carla.Location(x=ego_vehicle['location'][0],
                                                                                                 y=ego_vehicle['location'][1],
                                                                                                 z=ego_vehicle['location'][2]),
                                                                         location2=carla.Location(x=stop_sign_info['trigger_volume_location'][0],
                                                                                                 y=stop_sign_info['trigger_volume_location'][1],
                                                                                                 z=stop_sign_info['trigger_volume_location'][2]))
                    print_debug(f"[debug] stop sign, distance_to_trigger_location = {distance_to_trigger_location}")
                    # if stop_sign_info['distance'] < STOP_SIGN_STOP_DISTANCE:
                    if do_ego_must_stop(ego_location=ego_vehicle['location'], 
                                        stop_sign_dict=stop_sign_info, 
                                        map=self.map):
                        signal_answer = f"The ego vehicle should stop because of the stop sign({object_tags})."
                        final_brake_flag = True
                        final_stop_flag = True
                    else:
                        if distance_to_trigger_location < max(STOP_SIGN_BRAKE_DISTANCE, ego_data['speed'] * BRAKE_INTERVAL):
                            final_brake_flag = True
                            # final_stop_flag = False
                            signal_answer = f"The ego vehicle should drive slowly and stop at the stop sign({object_tags})."
                        else:
                            signal_answer = f"The ego vehicle should continue driving in the current lane and stop at the stop sign({object_tags})."
        elif traffic_light_info is not None:
            ego_distance_to_junction = ego_data.get('distance_to_junction', INF_MAX)
            if ego_distance_to_junction is None:
                ego_distance_to_junction = INF_MAX
            if flags['affected_by_red_light'] is True and not ego_data['is_in_junction']:
                print_debug(f"[debug] the vehicle is affected_by_red_light and is_in_junction")
                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=traffic_light_info)
                if len(object_tags) > 0:
                    if ego_distance_to_junction < RED_LIGHT_STOP_DISTANCE or traffic_light_info['distance'] < RED_LIGHT_STOP_RADIUS:
                        signal_answer = f"The ego vehicle should stop because of the traffic light({object_tags}) that is red."
                        final_brake_flag = True
                        final_stop_flag = True
                        print_debug(f"[debug] the vehicle should stop due to red light")
                    else:
                        if ego_distance_to_junction < max(RED_LIGHT_BRAKE_DISTANCE, ego_data['speed'] * BRAKE_INTERVAL):
                            final_brake_flag = True
                            # final_stop_flag = False
                            signal_answer = f"The ego vehicle should drive slowly in the current lane and stop at the intersection " +\
                                            f"because of the traffic light({object_tags}) that is red."
                            print_debug(f"[debug] the vehicle should brake due to red light")
                        else:
                            signal_answer = f"The ego vehicle should continue driving in the current lane and stop at the intersection " +\
                                            f"because of the traffic light({object_tags}) that is red."
            elif flags['affected_by_yellow_light'] is True and not ego_data['is_in_junction']:
                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=traffic_light_info)
                if len(object_tags) > 0:
                    if ego_distance_to_junction < RED_LIGHT_STOP_DISTANCE or traffic_light_info['distance'] < RED_LIGHT_STOP_RADIUS:
                        signal_answer = f"The ego vehicle should slow down because of the traffic light({object_tags}) that is yellow."
                        final_brake_flag = True
                        final_stop_flag = True
                    else:
                        if ego_distance_to_junction < max(RED_LIGHT_BRAKE_DISTANCE, ego_data['speed'] * BRAKE_INTERVAL):
                            final_brake_flag = True
                            # final_stop_flag = False
                            signal_answer = f"The ego vehicle should drive slowly in the current lane and stop at the intersection " +\
                                            f"because of the traffic light({object_tags}) that is yellow."
                        else:
                            signal_answer = f"The ego vehicle should continue driving in the current lane and stop at the intersection " +\
                                            f"because of the traffic light({object_tags}) that is yellow."
            # if stop_sign_info and stop_sign_info['affects_ego'] and stop_sign_info['distance'] < STOP_SIGN_CONSIDER_RAIUS:
            #     if already_stopped_at_stop_sign or self.stopped_at_stop_sign:
            #         final_brake_flag = False
            #         final_stop_flag = False
            #         answer += " The ego vehicle needn't stop at the stop sign now because it has already stopped at it."
            #     else:
            #         object_tags = self.get_key_of_key_object(key_object_infos, object_dict=stop_sign_info)
            #         answer = f"The ego vehicle should slow down and stop at the stop sign({object_tags})."
            #         final_brake_flag = True
            #         final_stop_flag = stop_sign_info['distance'] < STOP_SIGN_STOP_DISTANCE
        
        if "needn't stop" in signal_answer:
            answer += " " + signal_answer
        elif signal_answer != "":
            if answer == "There is no reason for the ego vehicle to brake.":
                answer = signal_answer
            else:
                answer = signal_answer + " " + answer
        
        if highway_merging_danger_flag or self.merging_and_needs_stop:
            answer = "The ego vehicle should control its speed to yield to other vehicles in the highway merging area."
            final_brake_flag = True
            final_stop_flag = False
            if isinstance(self.merging_danger_vehicles, list) and len(self.merging_danger_vehicles) > 0:
                self.merging_danger_vehicles.sort(key=lambda x: x['distance'])
                merge_vehicle = self.merging_danger_vehicles[0]
                color = get_vehicle_color(merge_vehicle)
                vehicletype = get_vehicle_type(merge_vehicle)
                rough_pos_str = get_rough_position(merge_vehicle)
                object_tags = self.get_key_of_key_object(key_object_infos, object_dict=merge_vehicle)
                answer = f"The ego vehicle must stop to yield the {color}{vehicletype}({object_tags}) that is " +\
                                                                    f"{rough_pos_str} in the merging area."
                final_brake_flag = True
                final_stop_flag = True
        if answer == "There is no reason for the ego vehicle to brake.":
            pass
        #     if abs(ego_data['speed']) <= 0.01:
        #         answer = "There is no reason for the ego vehicle to brake because ego vehicle is already static."

        # if must_wait_for_lane_change_str is not None:
        #     print_debug(f"[debug] must wait, frame_number = {self.current_measurement_index}")
        #     answer = must_wait_for_lane_change_str

        if must_brake_str is not None and final_brake_flag == False and final_stop_flag == False:
            answer = must_brake_str
            final_brake_flag = True
        if must_stop_str is not None:
            answer = must_stop_str
            final_brake_flag = True
            final_stop_flag = True

        self.answer43_brake = answer
        
        self.add_qas_questions(qa_list=qas_conversation_ego,
                            qid=8, 
                            chain=6,
                            layer=0,
                            qa_type='planning',
                            connection_up=-1,
                            connection_down=[(1,1), (2,2), (3,5), (3,6), (3,7), (3,8), (4,3), (5,0)],
                            question=question,
                            answer=answer,
                            object_tags=object_tags)
        
        return final_brake_flag, final_stop_flag, suggest_stop_flag

    def determine_ego_action_based_on_actor(actor, actor_type, ego_speed, ego_vehicle, qas_conversation_ego, 
                                            stop_signs, object_tags):
        """
        Answers "What should the ego vehicle do based on the {actor_type}?".

        Args:
            actor (dict): A dictionary containing information about the actor (traffic light or stop sign).
            actor_type (str): The type of actor ('traffic light' or 'stop sign').
            ego_speed (float): The current speed of the ego vehicle.
            ego_vehicle (dict): A dictionary containing information about the ego vehicle.
            measurements (dict): A dictionary containing sensor measurements.
            qas_conversation_ego (list): A list of dictionaries representing the conversation.

        Returns:
            None
        """

        already_stopped_at_stop_sign = False
        question = f"What should the ego vehicle do based on the {actor_type}?"
        sign_name = actor_type
        sign_behaviour = None
        # Check if the actor is present
        if actor is None:
            if actor_type == 'traffic light':
                answer = f"There is no {actor_type} affecting the ego vehicle."
                sign_name = None
                sign_behaviour = None
            elif actor_type == 'stop_sign':
                dist_thres = CARLA_STOP_SIGN_DISTANCE_THRESHOLD if self.in_carla else STOP_SIGN_DISTANCE_THRESHOLD
                cleared_stop_signs = [x for x in stop_signs if x['distance'] < dist_thres
                                                            and not x['affects_ego'] 
                                                            and x['position'][0] > STOP_SIGN_AHEAD_THRESHOLD]
                if cleared_stop_signs:
                    answer = f"The ego vehicle was affected by a stop sign({object_tags}), which has already been cleared."
                    sign_name = "stop sign which has been cleared"
                    sign_behaviour = "should speed up to normal speed"
                else:
                    answer = f"There is no {actor_type} affecting the ego vehicle."
                    return None, None, False
            else:
                answer = f"There is no {actor_type} affecting the ego vehicle."
                return None, None, False
        elif actor_type in ['traffic light', 'stop sign']:
            answer = f"The ego vehicle should follow the {actor_type}({object_tags})."
            sign_behaviour = f"should follow the {actor_type}({object_tags})"
            distances = [10, 15, 20, 40]

            if actor_type in ['traffic light']:
                # for traffic_light
                # 0 - Red; 1 - Yellow; 2 - Green; 3 - Off; 4 - Unknown;
                if actor['state'] == 0:
                    actor['state_str'] = 'Red'
                if actor['state'] == 1:
                    actor['state_str'] = 'Yellow'
                if actor['state'] == 2:
                    actor['state_str'] = 'Green'
                if actor['state'] == 3:
                    actor['state_str'] = 'Off'
                if actor['state'] == 4:
                    actor['state_str'] = 'Unknown'
            # Determine the action based on the ego vehicle's speed for red and green states
            if ego_speed > 5:
                red_str_speed = f'slow down and stop at the {actor_type}({object_tags})' 
            else: 
                red_str_speed = f'stop at the {actor_type}({object_tags})'
                
            if ego_speed < 5:
                green_str_speed = 'accelerate'
            else:
                green_str_speed = 'maintain its speed'

            for dist in distances:
                # Initialize the actor's state string if it doesn't exist
                if 'state_str' not in actor:
                    actor['state_str'] = ''

                # Check if the actor is within the current distance
                if actor['distance'] <= dist:
                    # Check if there is a leading vehicle affecting the ego vehicle
                    if ego_vehicle[f'hazard_detected_{dist}']:
                        # Handle the case where there is a leading vehicle
                        if actor_type == 'traffic light':
                            # Handle different traffic light states
                            if actor['state_str'] == 'Green':
                                answer = f"Based on the green traffic light({object_tags}) the ego vehicle can " +\
                                    f"{green_str_speed} and continue driving but should pay attention to the " +\
                                    f"vehicle in front and adjust its speed accordingly."
                                sign_behaviour = f"can " +\
                                    f"{green_str_speed} and continue driving but should pay attention to the " +\
                                    f"vehicle in front and adjust its speed accordingly"
                            elif actor['state_str'] == 'Yellow':
                                answer = f"The ego vehicle should slow down and prepare to stop at the " +\
                                                                                            f"traffic light({object_tags})."
                                sign_behaviour = f"should slow down and prepare to stop at the " +\
                                                                                            f"traffic light({object_tags})"
                            elif actor['state_str'] == 'Red':
                                answer = f"The ego vehicle should {red_str_speed} and stay behind other " +\
                                                                f"vehicles that are standing at the red light({object_tags})."
                                sign_behaviour = f"should {red_str_speed} and stay behind other " +\
                                                                f"vehicles that are standing at the red light({object_tags})"
                            else:
                                answer = f"The ego vehicle should follow the traffic light({object_tags})."
                                sign_behaviour = f"should follow the traffic light({object_tags})"
                        else:
                            # Handle the stop sign case
                            answer = f"The ego vehicle should {red_str_speed} and stay behind other vehicles " +\
                                                                            f"that are standing at the stop sign({object_tags})."
                            sign_behaviour = f"should {red_str_speed} and stay behind other vehicles " +\
                                                                            f"that are standing at the stop sign({object_tags})"
                    else:
                        # Handle the case where there is no leading vehicle
                        if actor_type == 'traffic light':
                            # Handle different traffic light states
                            if actor['state_str'] == 'Green':
                                answer = f"The ego vehicle can {green_str_speed} and continue driving because " +\
                                                                                    f"the traffic light({object_tags}) is green."
                                sign_behaviour = f"can {green_str_speed} and continue driving because " +\
                                                                                    f"the traffic light({object_tags}) is green"
                            elif actor['state_str'] == 'Yellow':
                                answer = f"The ego vehicle should slow down and prepare to stop at the " +\
                                                                                                f"traffic light({object_tags})."
                                sign_behaviour = f"should slow down and prepare to stop at the " +\
                                                                                                f"traffic light({object_tags})"
                            elif actor['state_str'] == 'Red':
                                answer = f"The ego vehicle should {red_str_speed}."
                                sign_behaviour = f"should {red_str_speed}"
                            else:
                                answer = f"The ego vehicle should follow the traffic light({object_tags})."
                                sign_behaviour = f"should follow the traffic light({object_tags})"
                        else:
                            # Handle the stop sign case
                            dist_thres = CARLA_STOP_SIGN_DISTANCE_THRESHOLD if self.in_carla else STOP_SIGN_DISTANCE_THRESHOLD
                            # distance_to_trigger_location = calculate_2D_distance(location1=carla.Location(x=ego_vehicle['location'][0],
                            #                                                                               y=ego_vehicle['location'][1],
                            #                                                                               z=ego_vehicle['location'][2]),
                            #                                                      location2=carla.Location(x=actor['trigger_volume_location'][0],
                            #                                                                               y=actor['trigger_volume_location'][1],
                            #                                                                               z=actor['trigger_volume_location'][2]))
                            if (((not self.in_carla and ego_speed < STOP_SIGN_SPEED_THRESHOLD) or \
                               (self.in_carla and ego_speed < CARLA_STOP_SIGN_SPEED_THRESHOLD)) and \
                               do_ego_must_stop(ego_location=ego_vehicle['location'], 
                                                stop_sign_dict=actor, 
                                                map=self.map)) or \
                               actor['position'][0] < -dist_thres:
                                answer = f"The ego vehicle can accelerate and continue driving if the " +\
                                            f"intersection is clear because it has already stopped at the stop sign({object_tags})."
                                sign_behaviour = f"can accelerate and continue driving if the " +\
                                            f"intersection is clear because it has already stopped at the stop sign({object_tags})"
                                already_stopped_at_stop_sign = True
                                self.stopped_at_stop_sign = True
                            else:
                                answer = f"The ego vehicle should {red_str_speed}."
                                sign_behaviour = f"should {red_str_speed}"
                    break # Break out of the loop since the actor has been handled
                else:
                    answer = f"The {actor_type}({object_tags}) is too far away to affect the ego vehicle."
                    sign_behaviour = f"maintain its speed and and wait until it gets closer to the traffic light({object_tags}) before making a decision"
        else:
            answer = f"The ego vehicle {self.traffic_sign_map[actor_name.replace(' ', '_')]['behaviour']}."
            sign_behaviour = f"{self.traffic_sign_map[actor_name.replace(' ', '_')]['behaviour']}"
        
            if actor_type in ['speed limit']:
                question = f"What should the ego vehicle do based on the speed limit sign?"
                if self.ahead_speed_limit is not None:
                    sign_name = f"limit sign ahead"
                    question = f"What should the ego vehicle do based on the speed limit sign ahead({object_tags}?"
                    answer = f"The ego vehicle should adjust its speed below {self.future_speed_limit} km/h soon."
                    sign_behaviour = f"should adjust its speed below {self.future_speed_limit} km/h soon"
                    if self.passed_speed_limit is not None:
                        question = f"What should the ego vehicle do based on the speed limit sign ahead({object_tags}) and passed?"
                        sign_name = f"speed limit sign ahead({object_tags}) and the speed limit sign passed"
                        answer = f"{answer[:-1]}, but remember that the passed {self.current_speed_limit} km/h " +\
                            "speed limit sign it still affecting the ego vehicle."
                        sign_behaviour = f"{sign_behaviour}, but remember that the passed {self.current_speed_limit} km/h " +\
                            f"speed limit sign it still affecting the ego vehicle, so current speed should not exceed {self.current_speed_limit} km/h"

        # Set the chain, layer, and connection values based on the actor type
        if actor_type == 'traffic light':
            chain = 2
            layer = 2
            connection_up = [(6,0)]
            connection_down = [(2,1)]
        else:
            chain = 1
            layer = 1
            connection_up = [(6,0)]
            connection_down = [(1,0)]

        # Add the question and answer to the conversation
        self.add_qas_questions(qa_list=qas_conversation_ego, 
                            qid=9, 
                            chain=chain, 
                            layer=layer, 
                            qa_type='planning',
                            connection_up=connection_up, 
                            connection_down=connection_down, 
                            question=question,
                            answer=answer,
                            object_tags=object_tags)
        
        return sign_name, sign_behaviour, already_stopped_at_stop_sign

    def determine_whether_ego_needs_to_change_lanes_due_to_obstruction(qas_conversation_ego,
                                                                        scenario_name,
                                                                        vehicles_by_id,
                                                                        static_objects,
                                                                        measurements,
                                                                        ego_data,
                                                                        important_objects, key_object_infos): 

        relevant_objects = []
        multiple_cones = False
        relevant_obj = None
        object_tags = []
        scenario_name = scenario_name.split('_')[0]
        change_flag = False
        change_dir = 0
        if ((self.current_measurement_index - self.last_special_move_index > self.scenario_ignore_interval) and \
           not (scenario_name in self.circumvent_scenarios and self.not_passed_circumvent_obstacles)) or \
           (self.obstacle_blocked_lane_id is not None and ego_data['lane_id'] != self.obstacle_blocked_lane_id):
            if 'InvadingTurn' not in scenario_name:
                scenario_name = 'Normal' # ignore scenarios

        change_lane_threshold = max(ego_data['speed'] * BRAKE_INTERVAL, CHANGE_LANE_THRESHOLD)
        # print_debug(f"[debug] scenario_name = {scenario_name}, frame_number = {self.current_measurement_index}") # 

        must_stop_str = None
        must_brake_str = None

        if 'ConstructionObstacle' in scenario_name:
            relevant_objects = [x for x in static_objects if x['type_id'] == 'static.prop.trafficwarning' 
                                                            and x['distance'] < change_lane_threshold
                                                            and x['position'][0] > 0.0 and is_vehicle_in_camera(self.CAMERA_FRONT, x)]
            if relevant_objects:
                self.role_vehicle_ids.append(relevant_objects[0]['id'])
        elif 'VehicleOpensDoorTwoWays' in scenario_name:
            if not self.in_carla:
                relevant_objects = [v for v in vehicles_by_id.values() if v['num_points'] >= MIN_OBJECT_NUM_POINT
                                    and v['type_id'] == 'vehicle.mercedes.coupe_2020' 
                                    and v['position'][0] > 0
                                    and v['position'][1] > BACK_CONSIDER_THRESHOLD
                                    and (float(v['distance']) < change_lane_threshold and v['speed'] < STOP_VEHICLE_SPEED) and is_vehicle_in_camera(self.CAMERA_FRONT, v)]
            elif self.role_actor is not None:
                relevant_objects = [v for v in vehicles_by_id.values() if v['num_points'] >= MIN_OBJECT_NUM_POINT
                                    and v['id'] == str(self.role_actor.id) 
                                    and v['position'][0] > 0
                                    and v['position'][1] > BACK_CONSIDER_THRESHOLD
                                    and (float(v['distance']) < change_lane_threshold and v['speed'] < STOP_VEHICLE_SPEED) and is_vehicle_in_camera(self.CAMERA_FRONT, v)]
            if relevant_objects:
                self.role_vehicle_ids.append(relevant_objects[0]['id'])
        elif 'InvadingTurn' in scenario_name:
            relevant_objects = list(filter(lambda x: x['type_id'] == 'static.prop.constructioncone' \
                                        and x['position'][0] >= INVADING_TURN_FORWARD_DISTANCE \
                                        and x['distance'] <= change_lane_threshold and is_vehicle_in_camera(self.CAMERA_FRONT, x), static_objects))
            print_debug(f"[debug] all cones: {[x['id'] for x in relevant_objects]}")
        elif 'ParkingExit' in scenario_name:
            if ego_data['lane_type_str'] == 'Parking':
                relevant_objects = [x for x in vehicles_by_id.values() if x['lane_type_str']=='Parking' 
                                    and x['position'][0] > 0]

        elif 'DynamicObjectCrossing' in scenario_name or 'ParkingCrossingPedestrian' in scenario_name or \
            'PedestrianCrossing' in scenario_name or 'VehicleTurningRoutePedestrian' in scenario_name:
            # in definition, DynamicObjectCrossing's obstacle might be a bicycleVehicleTurningRoute
            # but it does not exist in b2d
            # in PedestrianCrossing, there are 3 pedestrians
            # but since relevant_objects just saves the neareset one
            # we only reserve 1 nearest pedestrian, but in description, we describe them as 3

            # TODO: in theory: DynamicObjectCrossing and ParkingCrossingPedestrian has direction attribute
            # which means that thay can be on left side, but I didn't write its rule.
            relevant_objects = [x for x in pedestrians if x['num_points'] >= MIN_PEDESTRIAN_NUM_POINT and 
                                0.0 < x['position'][0] < change_lane_threshold
                                and (abs(x['speed']) > PEDESTRIAN_MIN_SPEED or x['distance'] < PEDESTRIAN_STOP_DISTANCE)
                                and (PEDESTRIAN_MIN_Y < x['position'][1] < PEDESTRIAN_MAX_Y or is_vehicle_in_camera(self.CAMERA_FRONT, x))] # carla lane width is 3.5m
        
        elif 'VehicleTurningRoute' == scenario_name:
            # bicycle length is roughly 1.6m
            relevant_objects = [x for x in vehicles_by_id.values() if x['num_points'] >= MIN_BICYCLE_NUM_POINT and
                                0.0 < x['position'][0] < change_lane_threshold
                                and (abs(x['speed']) > BICYCLE_MIN_SPEED or x['distance'] < BICYCLE_STOP_DISTANCE)
                                and ((BICYCLE_MIN_Y < x['position'][1] < BICYCLE_MAX_Y) or is_vehicle_in_camera(self.CAMERA_FRONT, x))
                                and 'bicycle' == x['base_type']] # carla lane width is 3.5m

        elif 'OppositeVehicleRunningRedLight' in scenario_name or 'OppositeVehicleTakingPriority' in scenario_name:
            # including OppositeVehicleTakingPriority and OppositeVehicleRunningRedLight
            # left -> right and right -> left all exists
            if not self.in_carla:
                relevant_objects = [x for x in vehicles_by_id.values() if abs(x['speed']) > PRIORITY_VEHICLE_MIN_SPEED
                                    and ('police' in x['type_id'] or 'ambulance' in x['type_id'] or 'firetruck' in x['type_id'])
                                    and ((x['approaching_dot_product'] < 0.0) or x['distance'] < TOO_CLOSE_THRESHOLD)]
            elif self.role_actor is not None:
                relevant_objects = [x for x in vehicles_by_id.values() if abs(x['speed']) > PRIORITY_VEHICLE_MIN_SPEED
                                    and x['id'] == str(self.role_actor.id)
                                    and ((x['approaching_dot_product'] < 0.0) or x['distance'] < TOO_CLOSE_THRESHOLD)]
            
        if self.current_measurement_index <= NEGLECT_FRAME_COUNT:
            # this is when background activity spawns vehicles around ego
            # speed of those vehicles can be wrong
            # so negelect
            relevant_objects = []

        # important object descriptions
        if relevant_objects:
            relevant_objects.sort(key=lambda x: x['distance'])
            relevant_obj = relevant_objects[0]

            rough_pos_str = f"{get_pos_str(relevant_obj['position'])} of the ego vehicle"

            # Determine the type of vehicle based on its type_id
            if 'ConstructionObstacle' in scenario_name:
                important_object_str = f'the construction warning {rough_pos_str}'
                category = "Traffic element"
                visual_description = "construction warning"
                del_object_in_key_info(key_object_infos, [relevant_obj])
            elif 'InvadingTurn' in scenario_name:
                multiple_cones = len(relevant_objects) > 1
                    
                plural = 's' if multiple_cones else ''
                important_object_str = f'the construction cone{plural} {rough_pos_str}'
                category = "Traffic element"
                visual_description = "construction cone"
                del_object_in_key_info(key_object_infos, relevant_objects)
            elif 'VehicleOpensDoorTwoWays' in scenario_name or 'ParkingExit' in scenario_name or \
                'OppositeVehicleTakingPriority' in scenario_name or 'OppositeVehicleRunningRedLight' in scenario_name:
                # Determine the color of the vehicle
                # color_str = relevant_obj["color_name"] + ' ' if relevant_obj.get("color_name") is not None \
                #                                             and relevant_obj["color_name"] != 'None' else ''
                if relevant_obj.get('color') is not None:
                    color_str = rgb_to_color_name(relevant_obj['color']) + ' '
                    if relevant_obj['color'] == [0, 28, 0] or relevant_obj['color'] == [12, 42, 12]:
                        color_str = 'dark green '
                    elif relevant_obj['color'] == [211, 142, 0]:
                        color_str = 'yellow '
                    elif relevant_obj['color'] == [145, 255, 181]:
                        color_str = 'blue '
                    elif relevant_obj['color'] == [215, 88, 0]:
                        color_str = 'orange '
                else:
                    color_str = ''

                category = "Vehicle"
                visual_description = f"{color_str}{relevant_obj['base_type']}"

                if 'VehicleOpensDoorTwoWays' in scenario_name and relevant_obj['distance'] <= VEHICLE_OPEN_DOOR_TRIGGER_DISTANCE:
                    # vehicle opens door at roughly 22m away
                    important_object_str = f"the {color_str}{relevant_obj['base_type']} with the open doors {rough_pos_str}"
                    cross_reason = "opens a door and blocks ego vehicle's lane"
                    cross_action = "keeps driving forward without decelerating"
                elif 'OppositeVehicleTakingPriority' in scenario_name or 'OppositeVehicleRunningRedLight' in scenario_name:
                    cross_reason = "is taking priority in the junction ahead"
                    cross_action = "keeps driving forward without waiting for it to pass"
                    if 'police' in relevant_obj['type_id']:
                        important_object_str = f'the {color_str}police car taking priority {rough_pos_str}'
                        visual_description = f"{color_str}police car"
                    elif 'ambulance' in relevant_obj['type_id']:
                        important_object_str = f'the {color_str}ambulance taking priority {rough_pos_str}'
                        visual_description = f"{color_str}ambulance"
                    elif 'firetruck' in relevant_obj['type_id']:
                        important_object_str = f'the {color_str}firetruck taking priority {rough_pos_str}'
                        visual_description = f"{color_str}firetruck"
                    else:
                        important_object_str = f'the {color_str}vehicle taking priority {rough_pos_str}'
                else:
                    important_object_str = f"the {color_str}{relevant_obj['base_type']} parking {rough_pos_str}"
                    cross_reason = "is parking in a place where blocks ego vehicle"
                    cross_action = "keeps driving forward without decelerating"
                
                old_str, _, _ = get_vehicle_str(relevant_obj)
                if is_object_in_key_object(key_object_infos, relevant_obj):
                # while old_str in important_objects:
                    important_objects.remove(old_str)
                del_object_in_key_info(key_object_infos, [relevant_obj])
                consider_vehicle([relevant_obj])
                set_vehicle_overlap(relevant_obj['id'], cross_reason, cross_action)
            
            elif 'DynamicObjectCrossing' in scenario_name or 'ParkingCrossingPedestrian' in scenario_name or 'VehicleTurningRoutePedestrian' in scenario_name:
                category = "Pedestrian"
                visual_description = "walking pedestrian"
                important_object_str = f'the pedestrian crossing the road {rough_pos_str}'
                del_object_in_key_info(key_object_infos, [relevant_obj])
                old_str = get_pedestrian_str(relevant_obj)
                while old_str in important_objects:
                    important_objects.remove(old_str)
            elif 'PedestrianCrossing' in scenario_name:
                category = "Pedestrian"
                visual_description = "3 pedestrians"
                important_object_str = f'3 pedestrians crossing the road {rough_pos_str}'
                # avoid repeat
                keys_to_remove = [k for k, v in key_object_infos.items() if v.get('category') == 'Pedestrian']
                for p in pedestrians:
                    old_str = get_pedestrian_str(p)
                    while old_str in important_objects:
                        important_objects.remove(old_str)
                for key in keys_to_remove:
                    del key_object_infos[key]
            elif 'VehicleTurningRoute' == scenario_name:
                category = "Vehicle"
                visual_description = 'bicycle'

                if relevant_obj.get('color') is not None:
                    color_str = rgb_to_color_name(relevant_obj['color']) + ' '
                    if relevant_obj['color'] == [0, 28, 0] or relevant_obj['color'] == [12, 42, 12]:
                        color_str = 'dark green '
                    elif relevant_obj['color'] == [211, 142, 0]:
                        color_str = 'yellow '
                    elif relevant_obj['color'] == [145, 255, 181]:
                        color_str = 'blue '
                    elif relevant_obj['color'] == [215, 88, 0]:
                        color_str = 'orange '

                important_object_str = f'the {color_str}bicycle crossing the road {rough_pos_str}'
                del_object_in_key_info(key_object_infos, [relevant_obj])
                cross_reason = "is crossing the road in front of the ego vehicle"
                cross_action = "keeps driving forward without waiting for it to pass"
                consider_vehicle([relevant_obj])
                set_vehicle_overlap(relevant_obj['id'], cross_reason, cross_action)
                old_str = get_bicycle_str(relevant_obj)
                for i in range(len(important_objects) - 1, -1, -1):
                    if old_str in important_objects[i]:
                        del important_objects[i]

            if scenario_name in ['ConstructionObstacle', 'ConstructionObstacleTwoWays', 'InvadingTurn', 
                                    'ParkingExit', 'VehicleOpensDoorTwoWays',
                                    'DynamicObjectCrossing', 'ParkingCrossingPedestrian', 'PedestrianCrossing',
                                    'VehicleTurningRoutePedestrian', 'VehicleTurningRoute',
                                    'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight'
                                    ]:
                # projected_points, projected_points_meters = project_all_corners(relevant_obj, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                # Generate a unique key and value for the vehicle object
                project_dict = get_project_camera_and_corners(relevant_obj, self.CAM_DICT)
                key, value = self.generate_object_key_value(
                    id=relevant_obj['id'],
                    category = category,
                    visual_description = visual_description,
                    detailed_description = important_object_str,
                    object_count = len(key_object_infos),
                    is_role=True,
                    obj_dict = relevant_obj,
                    projected_dict=project_dict
                )
                key_object_infos[key] = value
                object_tags = [key]
                important_objects.append(f"{important_object_str}({object_tags})")

            else:
                important_objects.append(important_object_str)

        

        question = "Does the ego vehicle need to change lanes or deviate from the lane center due to an "\
                    "upcoming obstruction?"
        answer = "No, the ego vehicle can stay on its current lane."
        change_flag = False

        question2 = 'Is there an obstacle on the current road?'
        answer2 = 'No, there is no obstacle on the current route.'
        
        in_scene = False
        if self.current_measurement_index > 5 and \
            scenario_name in ['Accident', 'AccidentTwoWays', 'ConstructionObstacle', 'ConstructionObstacleTwoWays',
                                'InvadingTurn', 'HazardAtSideLane', 'HazardAtSideLaneTwoWays', 'ParkedObstacle',
                                'ParkedObstacleTwoWays', 'VehicleOpensDoorTwoWays', 
                                'DynamicObjectCrossing', 'ParkingCrossingPedestrian', 'PedestrianCrossing', 'VehicleTurningRoutePedestrian', 'VehicleTurningRoute',
                                'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight']:

            obstacle = {'Accident': 'accident',
                        'AccidentTwoWays': 'accident', 
                        'ConstructionObstacle': 'construction warning', 
                        'ConstructionObstacleTwoWays': 'construction warning', 
                        'InvadingTurn': 'invading vehicles on the opposite lane', 
                        'HazardAtSideLane': 'two bicycles',
                        'HazardAtSideLaneTwoWays': 'two bicycles',
                        'ParkedObstacle': 'parked vehicle', 
                        'DynamicObjectCrossing': 'walking pedestrian',
                        'ParkingCrossingPedestrian': 'walking pedestrian',
                        'VehicleTurningRoutePedestrian': 'walking pedestrian',
                        'PedestrianCrossing': '3 walking pedestrians',
                        'VehicleTurningRoute': 'bicycle',
                        'OppositeVehicleTakingPriority': 'running vehicle',
                        'OppositeVehicleRunningRedLight': 'running vehicle',
                        'ParkedObstacleTwoWays': 'parked vehicle', 
                        'VehicleOpensDoorTwoWays': 'vehicle with the opened door'}[scenario_name]
            
            in_scene = True
            # changed_route = measurements["changed_route"]
            if object_tags and len(object_tags) > 0:
                obstacle = f"{obstacle}({object_tags})"
            
            if 'HazardAtSideLane' in scenario_name:
                relevant_objects = [v for v in vehicles_by_id.values() if v['base_type'] == 'bicycle' 
                                                                    and self.should_consider_vehicle(v) 
                                                                    and float(v['distance']) < change_lane_threshold
                                                                    and 0 <= v['lane_relative_to_ego'] <= 1]
                if len(relevant_objects) == 1:
                    obstacle = f'bicycle({object_tags})'
            elif scenario_name not in ['VehicleOpensDoorTwoWays', 'ConstructionObstacle', 
                                        'ConstructionObstacleTwoWays', 'InvadingTurn',
                                        'DynamicObjectCrossing', 'ParkingCrossingPedestrian', 'PedestrianCrossing', 'VehicleTurningRoutePedestrian', 'VehicleTurningRoute',
                                        'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight']:
                # Circuvment to static vehicle scenarios.
                # Accident
                # AccidentTwoWays
                # ParkedObstacle
                # ParkedObstacleTwoWays
                # print_debug(f"[debug] vehicles_by_id.values(): {vehicles_by_id.values()}") # 
                # print_debug(f"[debug] self.role_actor.id = {self.role_actor.id}") # 
                if self.in_carla:
                    if self.role_actor is None:
                        relevant_objects = []
                    relevant_objects = [v for v in vehicles_by_id.values() if str(self.role_actor.id) == v['id']
                                                                            and self.should_consider_vehicle(v)
                                                                            and float(v['distance']) < change_lane_threshold]
                else:
                    relevant_objects = [v for v in vehicles_by_id.values() if self.should_consider_vehicle(v) 
                                                                        and abs(v['speed']) <= STOP_VEHICLE_SPEED
                                                                        and float(v['distance']) < change_lane_threshold]
                # print_debug(f"[debug] raw relevant_objects condition 1: {[v['type_id'] for v in vehicles_by_id.values() if self.should_consider_vehicle(v)]}") #
                # print_debug(f"[debug] raw relevant_objects condition 2: {[v['type_id'] for v in vehicles_by_id.values() if abs(v['speed']) <= 0.001]}") #
                # print_debug(f"[debug] raw relevant_objects condition 3: {[v['type_id'] for v in vehicles_by_id.values() if float(v['distance']) < CHANGE_LANE_THRESHOLD]}") #


                if (not self.in_carla) and ('Accident' in scenario_name):
                    relevant_objects = [v for v in relevant_objects if 'police' in v['type_id']]
            
            relevant_objects.sort(key = lambda x: float(x['distance']))
            # print_debug(f"[debug] relevant_objects: {relevant_objects}") # 

            if relevant_objects:
                if 'Accident' in scenario_name: 
                    object_tags = [k for k, v in key_object_infos.items() if 'police' in v['Visual_description']]
                    # print_debug(f"[debug] object_tag: {object_tags}") # 
                    if object_tags and len(object_tags) > 0:
                        police_car = key_object_infos[object_tags[0]]
                        cross_reason = "is parked on the ego vehicle's lane"
                        cross_action = "keeps driving forward in the current lane"
                        self.role_vehicle_ids.append(police_car['id'])
                        consider_vehicle([police_car])
                        set_vehicle_overlap(police_car['id'], cross_reason, cross_action)  
                        
                elif 'HazardAtSideLane' in scenario_name: 
                    object_tags = [k for k, v in key_object_infos.items() if 'bicycle' in v['Visual_description']]    
                elif 'ParkedObstacle' in scenario_name:
                    assert len(relevant_objects), relevant_objects

                    relevant_obj = relevant_objects[0]
                    # projected_points, projected_points_meters = project_all_corners(relevant_obj, 
                    #                                                                 self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                    # if projected_points is not None and len(projected_points) > 0:
                    #     two_d_box = self.generate_2d_box_from_projected_points(projected_points)
                    keys = [k for k, v in key_object_infos.items() if relevant_obj['id']==v['id']]
                    assert len(keys)==1, keys
                    object_tags = keys
                    self.role_vehicle_ids.append(key_object_infos[keys[0]]['id'])
                    cross_reason = "is parked on the ego vehicle's lane"
                    cross_action = "keeps driving forward in the current lane"
                    consider_vehicle([key_object_infos[keys[0]]])
                    set_vehicle_overlap(key_object_infos[keys[0]]['id'], cross_reason, cross_action)
                    
                elif 'VehicleOpensDoorTwoWays' in scenario_name:
                    assert len(relevant_objects), relevant_objects

                    relevant_obj = relevant_objects[0]
                    self.role_vehicle_ids.append(relevant_obj['id'])
                    # projected_points, projected_points_meters = project_all_corners(relevant_obj, 
                    #                                                                 self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                    # if projected_points is not None and len(projected_points) > 0:
                    #     two_d_box = self.generate_2d_box_from_projected_points(projected_points)
                    keys = [k for k, v in key_object_infos.items() if relevant_obj['id']==v['id']]

                    object_tags = keys
                
                changed_route = relevant_objects[0]['lane_id'] != ego_data['lane_id']
                if changed_route and relevant_obj and \
                    scenario_name not in ['Normal', 'DynamicObjectCrossing', 'ParkingCrossingPedestrian', 'PedestrianCrossing', 'VehicleTurningRoutePedestrian', 'VehicleTurningRoute',
                                        'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'VehicleOpensDoorTwoWays']:
                    if 'InvadingTurn' == scenario_name:
                        answer = f"The ego vehicle should deviate slightly to the right from its current lane to avoid the {obstacle}."
                        # DON'T MODIFY "deviate slightly" here because this is used for recognition in ego behaviour answer_behaviour_questions
                        change_flag = True
                        answer2 = f'Yes, there might be invading vehicles from the opposite lane on the current road.'
                    else:
                        # print("[debug] TODO: line 1463: lane_change related, has to be implemented.")
                        # route_start = np.array(measurements['route_original'][0])
                        # route_end = np.array(measurements['route_original'][1])
                        
                        # route_vector = route_end - route_start
                        # ego_to_route_start = route_start  # Assuming ego vehicle is at [0, 0]

                        # # Calculate the projection of ego_to_route_start onto route_vector
                        # projection_length = np.dot(route_vector, ego_to_route_start) / np.linalg.norm(route_vector)
                        
                        # # Calculate lateral distance using Pythagorean theorem
                        # distance_to_route_start = np.linalg.norm(ego_to_route_start)
                        # lateral_distance = np.sqrt(distance_to_route_start**2 - projection_length**2)

                        lateral_distance = 3.5

                        # usually roads in carla are 3.5 wide
                        changing_or_has_changed = "has already changed" if lateral_distance > CARLA_LANE_WIDTH / 2. else "is "\
                                                                                                        "changing"
                        answer = f"The ego vehicle {changing_or_has_changed} to another lane to "\
                                    f"circumvent the {obstacle}."
                        change_flag = False if lateral_distance > CARLA_LANE_WIDTH / 2. else True
                else:
                    keywords = ['Accident', 'ConstructionObstacle', 'HazardAtSideLane', 'ParkedObstacle']
                    if any(keyword in scenario_name for keyword in keywords):
                        if ego_data['lane_change'] == 1:
                            side = 'the right lane'
                        elif ego_data['lane_change'] == 2:
                            side = 'the left lane'
                        elif ego_data['lane_change'] == 3:
                            side = 'either side'
                        
                        if ego_data['lane_change'] in [1, 2, 3]:
                            answer = f"The ego vehicle must change to {side} to circumvent the {obstacle}."
                            change_flag = True

                            if not obstacle.startswith('two'):
                                obstacle2 = 'an '+obstacle if obstacle[0] in ['a', 'e', 'i', 'o', 'u'] else 'a '+obstacle
                            else:
                                obstacle2 = obstacle
                            obstacle2 = 'are '+obstacle2 if obstacle2.startswith('two') else 'is '+obstacle2
                            answer2 = f'Yes, there {obstacle2} on the current road.'
                        elif ego_data['lane_change'] in [0]:
                            if 'TwoWays' in scenario_name:
                                answer = f"The ego vehicle must change to the opposite lane to circumvent the {obstacle}."
                                change_flag = True

                                if not obstacle.startswith('two'):
                                    obstacle2 = 'an '+obstacle if obstacle[0] in ['a', 'e', 'i', 'o', 'u'] else 'a '+obstacle
                                else:
                                    obstacle2 = obstacle
                                obstacle2 = 'are '+obstacle2 if obstacle2.startswith('two') else 'is '+obstacle2
                                answer2 = f'Yes, there {obstacle2} on the current road.'
                            else:
                                answer = f"No, the ego vehicle can stay on its current lane. But the ego vehicle must stop because there's no way to circumvent the {obstacle}."
                                must_stop_str = f"The ego vehicle must stop because there's no way to circumvent the {obstacle}."
                                change_flag = False

                                obstacle2 = 'an '+obstacle if obstacle[0] in ['a', 'e', 'i', 'o', 'u'] else 'a '+obstacle
                                obstacle2 = 'are '+obstacle2 if obstacle2.startswith('two') else 'is '+obstacle2
                                answer2 = f'Yes, there {obstacle2} on the current road.'

                    elif 'VehicleOpensDoor' in scenario_name:
                            if relevant_obj['distance'] > VEHICLE_OPEN_DOOR_TRIGGER_DISTANCE:
                                answer = "No, the ego vehicle can stay on its current lane."
                                change_flag = False
                                answer2 = 'No, there is no obstacle on the current route.'
                            else:
                                lane_str = 'opposite lane' if ego_data['lane_change'] in [0, 1] else 'left lane'
                                answer = f"The ego vehicle must change to the {lane_str} to circumvent the {obstacle}."
                                change_flag = True
                                if ego_data['lane_change'] == 3:
                                    ego_data['lane_change'] = 2 # cannot go right because the opened door vehicle is there.

                                if not obstacle.startswith('two'):
                                    obstacle2 = 'an '+obstacle if obstacle[0] in ['a', 'e', 'i', 'o', 'u'] else 'a '+obstacle
                                else:
                                    obstacle2 = obstacle
                                obstacle2 = 'are '+obstacle2 if obstacle2.startswith('two') else 'is '+obstacle2
                                answer2 = f'Yes, there {obstacle2} on the current road.'
                    elif scenario_name == 'InvadingTurn':
                        answer = f"The ego vehicle must shift slightly to the right side to avoid {obstacle}."
                        change_flag = True

                        answer2 = f'Yes, there might be invading vehicles from the opposite lane on the current road.'
                    # 'AccidentTwoWays', 'ConstructionObstacleTwoWays', 'HazardAtSideLaneTwoWays', 
                    # 'ParkedObstacleTwoWays', 'VehicleOpensDoorTwoWays'
                    elif 'DynamicObjectCrossing' in scenario_name or 'ParkingCrossingPedestrian' in scenario_name or 'VehicleTurningRoutePedestrian' in scenario_name:
                        answer = f"No, the ego vehicle can stay on its current lane. But the ego vehicle must stop because there's a pedestrian({object_tags}) crossing the road."
                        if relevant_obj['speed'] < PEDESTRIAN_MIN_SPEED:
                            must_brake_str = f"The ego vehicle must brake because there's a pedestrian({object_tags}) which is about to cross the road."
                        else:
                            must_stop_str = f"The ego vehicle must stop because there's a pedestrian({object_tags}) crossing the road."
                        change_flag = False
                        answer2 = f'Yes, there is a pedestrian({object_tags}) crossing the road.'
                    elif 'PedestrianCrossing' in scenario_name:
                        answer = f"No, the ego vehicle can stay on its current lane. But the ego vehicle must stop because there are 3 pedestrians({object_tags}) crossing the road."
                        if relevant_obj['speed'] < PEDESTRIAN_MIN_SPEED:
                            must_brake_str = f"The ego vehicle must brake because there are 3 pedestrians({object_tags}) which is about to cross the road."
                        else:
                            must_stop_str = f"The ego vehicle must stop because there are 3 pedestrians({object_tags}) crossing the road."
                        change_flag = False
                        answer2 = f'Yes, there are 3 pedestrians({object_tags}) crossing the road.'
                    elif scenario_name == 'VehicleTurningRoute':
                        answer = f"No, the ego vehicle can stay on its current lane. But the ego vehicle must stop because there's a bicycle({object_tags}) crossing the road."
                        if relevant_obj['speed'] < BICYCLE_MIN_SPEED:
                            must_brake_str = f"The ego vehicle must brake because there's a bicycle({object_tags}) which is about to cross the road."
                        else:
                            must_stop_str = f"The ego vehicle must stop because there's a bicycle({object_tags}) crossing the road."
                        change_flag = False
                        answer2 = f'Yes, there is a bicycle({object_tags}) crossing the road.'
                    elif 'OppositeVehicleTakingPriority' in scenario_name or 'OppositeVehicleRunningRedLight' in scenario_name:
                        obstacle2 = 'a vehicle'
                        if 'police' in relevant_obj['type_id']:
                            obstacle2 = 'a police car'
                        elif 'ambulance' in relevant_obj['type_id']:
                            obstacle2 = 'an ambulance'
                        elif 'firetruck' in relevant_obj['type_id']:
                            obstacle2 = 'a firetruck'
                        obstacle2 = obstacle2 + f"({object_tags})"
                        answer = f"No, the ego vehicle can stay on its current lane. But the ego vehicle must stop because there's {obstacle2} taking priority."
                        must_stop_str = f"The ego vehicle must stop because there's {obstacle2} taking priority."
                        change_flag = False
                        answer2 = f'Yes, there is {obstacle2} taking priority.'
                    else: 
                        answer = f"The ego vehicle must change to the opposite lane to circumvent the {obstacle}."
                        change_flag = True
                        obstacle2 = 'an '+obstacle if obstacle[0] in ['a', 'e', 'i', 'o', 'u'] else 'a '+obstacle
                        obstacle2 = 'are '+obstacle2 if obstacle2.startswith('two') else 'is '+obstacle2
                        answer2 = f'Yes, there {obstacle2} on the current road.'
    
            # if changed_route \
            #         and answer == "No, the ego vehicle can stay on its current lane." \
            #         and scenario_name != 'ParkingExit':
            #     answer = "The ego vehicle must change back to the original lane after passing the obstruction."

        elif 'ParkingExit' in scenario_name:
            if ego_data['lane_type_str'] == 'Parking':
                if ego_data['right_lane_marking_color_str'] == 'White' and ego_data['right_lane_marking_type_str'] == 'Solid':
                    # print_debug(f"[debug] right, index = {self.current_measurement_index}")
                    ego_data['lane_change'] = 1
                if ego_data['left_lane_marking_color_str'] == 'White' and ego_data['left_lane_marking_type_str'] == 'Solid':
                    # print_debug(f"[debug] left, index = {self.current_measurement_index}")
                    ego_data['lane_change'] = 2
                if ego_data['lane_change'] == 0:
                    # print_debug(f"[debug] no direction! index = {self.current_measurement_index}")
                    ego_data['lane_change'] = 2
                if ego_data['lane_change'] == 2:
                    side = 'the left lane'
                elif ego_data['lane_change'] == 1:
                    side = 'the right lane'
                elif ego_data['lane_change'] == 3:
                    side = 'either side'
                answer = f"The ego vehicle must change to {side} to exit the parking space."
                change_flag = True
        
        # print_debug(f"[debug] {self.current_measurement_path}: raw, in_scene = {in_scene}, change_flag = {change_flag}, last_left_lane = {self.last_left_lane}, ego_lane_id = {ego_data['lane_id']}")
        if in_scene and change_flag:
            # self.last_special_move_index = self.current_measurement_index
            # self.last_left_lane = ego_data['lane_id']
            # the logic here is moved to generate_ego_vehicle_actions
            # because the logic identifying if the ego vehicle could change lane or not is there.
            pass

        print_debug(f"[debug] change_flag = {change_flag}, self.obstacle_blocked_lane_id = {self.obstacle_blocked_lane_id}")
        if change_flag and self.obstacle_blocked_lane_id is None:
            print_debug(f"[debug] ego_data['lane_id'] = {ego_data['lane_id']}")
            self.obstacle_blocked_lane_id = ego_data['lane_id']

        print_debug(f"[debug] in ego_actions, self.obstacle_blocked_lane_id = {self.obstacle_blocked_lane_id}")

        if change_flag == False:
            if self.last_left_lane == ego_data['lane_id'] and scenario_name in self.circumvent_scenarios:
                answer = "The ego vehicle is still changing lane to circumvent the obstacle."
                change_flag = True
            if self.last_left_lane != ego_data['lane_id']:
                self.last_left_lane = INVALID_NUM # clean mark
        
        # print_debug(f"[debug] {self.current_measurement_path}: after, in_scene = {in_scene}, change_flag = {change_flag}, last_left_lane = {self.last_left_lane}, ego_lane_id = {ego_data['lane_id']}, answer = {answer}")

        self.add_qas_questions(qa_list=qas_conversation_ego,
                                qid=10,
                                chain=3,
                                layer=8,
                                qa_type='planning',
                                connection_up=[(6, 0)] ,
                                connection_down=[(3, 9)],
                                question=question,
                                answer=answer,
                                object_tags=object_tags)
        
        self.add_qas_questions(qa_list=qas_conversation_ego,
                                qid=11,
                                chain=3,
                                layer=9,
                                qa_type='perception',
                                connection_up=[(3,8)],
                                connection_down=-1,
                                question=question2,
                                answer=answer2,
                                object_tags=object_tags)
    
        for object_tag in object_tags:
            key_object_infos[object_tag]['is_role'] = True
        change_dir = ego_data['lane_change']
        if change_dir in [0]: change_dir = 2 # change left to opposite lane
        if change_dir in [0, 1] and 'VehicleOpensDoor' in scenario_name: change_dir = 2 # change left to opposite lane
        return change_flag, change_dir, answer, answer2, object_tags, must_stop_str, must_brake_str
    
    def determine_whether_ego_needs_to_change_lanes_due_to_other_factor(qas_conversation_ego,
                                                                    scenario_name,
                                                                    vehicles_by_id,
                                                                    static_objects,
                                                                    measurements,
                                                                    ego_data, important_objects, key_object_infos):
            
        relevant_objects = []
        object_tags = []
        scenario_name = scenario_name.split('_')[0]
        change_flag = False
        change_dir = 0

        question = "Does the ego vehicle need to change lanes or deviate from the lane " +\
                "for reasons other than the upcoming obstruction? Why?"
        answer = "No, no other reason supports the ego vehicle to change lane."

        change_lane_on_highway = False
        yielding_emergency = False
        # add vehicles in rear on target lane into key objects\
        # and consider crossing as well
        x = (measurements['x_command_far'] - measurements['x'])**2
        y = (measurements['y_command_far'] - measurements['y'])**2
        command_distance = np.sqrt(x + y)
        if measurements['command_near'] == 5 or (measurements['command_far'] == 5 and command_distance < 15.0):
            if ego_data['lane_change'] in [2, 3]: # can do left or both
                change_flag = True
                change_dir = 2
                answer = "Yes, the current command orders the ego vehicle to change to left lane soon."
                # DO NOT modify "current command orders", because it is used for identification in behaviour.py
        if measurements['command_near'] == 6 or (measurements['command_far'] == 6 and command_distance < 15.0): 
            if ego_data['lane_change'] in [1, 3]: # can do right or both
                change_flag = True
                change_dir = 1
                answer = "Yes, the current command orders the ego vehicle to change to right lane soon."
                # DO NOT modify "current command orders", because it is used for identification in behaviour.py
        
        self.not_passed_circumvent_obstacles = False
        self.distance_to_circumvent_obstacle = None
        role_dicts = []
        if len(self.role_vehicle_ids) > 0 or self.role_actor is not None or self.blocker_actor is not None:
            self.distance_to_circumvent_obstacle = INF_MAX
            role_dicts = [x for x in vehicles_by_id.values() if x['id'] in self.role_vehicle_ids or
                                                                (self.role_actor is not None and x['id'] == str(self.role_actor.id)) or
                                                                (self.blocker_actor is not None and x['id'] == str(self.blocker_actor.id))]
            static_dicts = [x for x in static_objects if x['id'] in self.role_vehicle_ids or
                                                         (self.role_actor is not None and x['id'] == str(self.role_actor.id)) or
                                                         (self.blocker_actor is not None and x['id'] == str(self.blocker_actor.id))]
            role_dicts.extend(static_dicts)
            role_dicts = [x for x in role_dicts if 'static.prop.warning' not in x['type_id']]
        for role in role_dicts:
            self.distance_to_circumvent_obstacle = min(self.distance_to_circumvent_obstacle, role['position'][0])
            print_debug(f"[debug] role actor id={role['id']}, type_id={role['type_id']}, position={role['position']}, distance={role['distance']}")
            if role['position'][0] > BACK_CONSIDER_THRESHOLD or \
              (role['position'][0] < 0 and role['distance'] < self.get_obstacle_pass_offset(scenario_name)):
                self.not_passed_circumvent_obstacles = True

        print_debug(f"[debug] updated self.not_passed_circumvent_obstacles, it is {self.not_passed_circumvent_obstacles} now.")

        if self.opposite_flag:
            dev_deg = ego_data['lane_deviant_degree']
            dev_str = ""
            if abs(dev_deg) < 150:
                if dev_deg < 0:
                    dev_str = "steer to the right then "

            print_debug(f"[debug] ego_data['opposite_and_right_front_clear'] = {ego_data['opposite_and_right_front_clear']}, not self.not_passed_circumvent_obstacles = {not self.not_passed_circumvent_obstacles}")
            if ego_data['opposite_and_right_front_clear'] and (not self.not_passed_circumvent_obstacles):
                answer = "Yes, the ego vehicle is driving in the opposite direction and " +\
                    f"must {dev_str}change back to the correct(right) lane since the lane is clear."
                change_flag = True
                print_debug(f"[debug] MUST CHANGE")
            else:
                answer = "Yes, the ego vehicle is driving in the opposite direction and " +\
                    f"must {dev_str}change back to the correct(right) lane as soon as possible. " +\
                    "But not now because the ego vehicle has to stay on its current lane temporarily and " +\
                    "accelerate to bypass the blocked session of the correct lane."
                # don't change 'But not now', it is useful in inference stage.
                change_flag = False
                print_debug(f"[debug] NOT YET")
            change_dir = 2
        elif 'YieldToEmergencyVehicle' in scenario_name:
            if not self.in_carla:
                relevant_objects = [x for x in vehicles_by_id.values() if -40.0 < x['position'][0] < 4.0
                                    and x['num_points'] >= 15
                                    and ('police' in x['type_id'] or 'ambulance' in x['type_id'] or 'firetruck' in x['type_id'])]
            elif self.role_actor is not None:
                relevant_objects = [x for x in vehicles_by_id.values() if -40.0 < x['position'][0] < 4.0
                                    and x['num_points'] >= 15
                                    and x['id'] == str(self.role_actor.id)]
            if relevant_objects:
                relevant_objects = sorted(relevant_objects, key=lambda x: x['distance'])
                relevant_obj = relevant_objects[0]
                if relevant_obj['lane_relative_to_ego'] != 0:
                    yielding_emergency = True

            relevant_objects = [x for x in relevant_objects if x['position'][0] < -1.0]

            if relevant_objects:
                relevant_obj = relevant_objects[0]
                rough_pos_str = get_pos_str(relevant_obj['position'])
                color_str = ""
                if relevant_obj.get('color') is not None:
                    color_str = rgb_to_color_name(relevant_obj['color']) + ' '
                    if relevant_obj['color'] == [0, 28, 0] or relevant_obj['color'] == [12, 42, 12]:
                        color_str = 'dark green '
                    elif relevant_obj['color'] == [211, 142, 0]:
                        color_str = 'yellow '
                    elif relevant_obj['color'] == [145, 255, 181]:
                        color_str = 'blue '
                    elif relevant_obj['color'] == [215, 88, 0]:
                        color_str = 'orange '

                category = "Vehicle"
                cross_reason = "is taking priority behind the ego vehicle"
                cross_action = "keeps driving in the current lane without shifting aside to yield it"
                visual_description = f"{color_str}{relevant_obj['base_type']}"
                if 'police' in relevant_obj['type_id']:
                    important_object_str = f'the {color_str}police car taking priority {rough_pos_str}'
                    visual_description = f"{color_str}police car"
                elif 'ambulance' in relevant_obj['type_id']:
                    important_object_str = f'the {color_str}ambulance taking priority {rough_pos_str}'
                    visual_description = f"{color_str}ambulance"
                elif 'firetruck' in relevant_obj['type_id']:
                    important_object_str = f'the {color_str}firetruck taking priority {rough_pos_str}'
                    visual_description = f"{color_str}firetruck"
                else:
                    important_object_str = f'the {color_str}vehicle taking priority {rough_pos_str}'

                
                # while old_str in important_objects:
                if is_object_in_key_object(key_object_infos, relevant_obj):
                    important_objects.remove(old_str)
                del_object_in_key_info(key_object_infos, [relevant_obj])

                consider_vehicle([relevant_obj])
                set_vehicle_overlap(relevant_obj['id'], cross_reason, cross_action)
                old_str, _, _ = get_vehicle_str(relevant_obj)
                
                # projected_points, projected_points_meters = project_all_corners(relevant_obj, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                project_dict = get_project_camera_and_corners(relevant_obj, self.CAM_DICT)
                # Generate a unique key and value for the vehicle object
                key, value = self.generate_object_key_value(
                    id=relevant_obj['id'],
                    category=category,
                    visual_description=visual_description,
                    detailed_description=important_object_str,
                    object_count=len(key_object_infos),
                    is_role=True,
                    obj_dict=relevant_obj,
                    projected_dict=project_dict
                )
                key_object_infos[key] = value
                
                important_object_str = important_object_str + f'({[key]})'
                important_objects.append(important_object_str)

                if relevant_obj['position'][0] < 0.0 and relevant_obj['lane_relative_to_ego'] == 0:
                    object_tags = [key]
                    change_flag = True
                    change_dir = ego_data['lane_change']
                    answer = f"Yes, the vehicle have to change lane to yield to emergency vehicle({object_tags}) approaching from behind."
        
        elif scenario_name in self.highway_change_lane_scenarios:
            # 'InterurbanAdvancedActorFlow' does not change lane
            at_leftmost = ego_data['lane_change'] in [0, 1]
            at_rightmost = ego_data['lane_change'] in [0, 2]
            if self.first_lane_command == 5 and at_leftmost is False:
                if scenario_name in ['InterurbanAdvancedActorFlow']:
                    answer = "Yes, the ego vehicle needs to enter the fast lane on the left."
                else:
                    answer = "Yes, the ego vehicle must change lanes to the far-left lane according to given command."
                    # DO NOT modify "must change lane" because it is used for identification in behaviour.py
                change_flag = True
                change_dir = 2
                change_lane_on_highway = True
            if self.first_lane_command == 6 and at_rightmost is False:
                if scenario_name in ['InterurbanAdvancedActorFlow']:
                    answer = "Yes, the ego vehicle needs to enter the fast lane on the right."
                else:
                    answer = "Yes, the ego vehicle must change lanes to the far-right lane according to given command."
                    # DO NOT modify "must change lane" because it is used for identification in behaviour.py
                change_flag = True
                change_dir = 1
                change_lane_on_highway = True

        special_change_lane_reason = None
        ood_flag = False

        if self.strict_mode:
            if change_dir == 0: # ood, and changing back to original lane after circumvented the obstacle
                if not ego_data['is_in_junction'] and ego_data.get('distance_to_junction'):
                    if ego_data['road_id'] == self.correct_road and ego_data['lane_id'] != self.correct_lane:
                        nearest_correct_location, nearest_forward_vector = get_nearest_point_on_lane(map=self.map,
                                                                            x=ego_data['location'][0],
                                                                            y=ego_data['location'][1],
                                                                            z=ego_data['location'][2],
                                                                            road_id=self.correct_road,
                                                                            lane_id=self.correct_lane)
                        if nearest_correct_location is not None:
                            ood_flag = True
                            relative_nearest_point = transform_to_ego_coordinates([nearest_correct_location.x,
                                                                                nearest_correct_location.y,
                                                                                nearest_correct_location.z],
                                                                                ego_data['world2ego'])
                            reject = ""
                            if relative_nearest_point[1] < 0.0: # left
                                change_flag = True
                                change_dir = 2
                                answer = "Yes, the ego vehicle is not on its target lane " +\
                                        "and it has to change to the correct lane on the left."
                                special_change_lane_reason = "change to the correct lane on the left"
                                if not ego_data['left_front_clear']:
                                    reject = " But not now because the left lane is blocked."
                                    change_dir = 0
                                    change_flag = False
                            else: # right
                                change_flag = True
                                change_dir = 1
                                answer = "Yes, the ego vehicle is not on its target lane " +\
                                        "and it has to change to the correct lane on the right."
                                special_change_lane_reason = "change to the correct lane on the right"
                                if not ego_data['right_front_clear']:
                                    reject = " But not now because the right lane is blocked."
                                    change_dir = 0
                                    change_flag = False
                            if yielding_emergency:
                                reject = " But not now because the ego vehicle is yielding the emergency vehicle."
                                change_dir = 0
                                change_flag = False
                            if self.not_passed_circumvent_obstacles:
                                reject = " But not now because the ego vehicle has not yet passed the obstacle in that lane."
                                change_dir = 0
                                change_flag = False
                            
                            answer = answer + reject
        
        if not ('ParkingExit' in scenario_name and ego_data['lane_type_str'] == 'Parking') and \
           not is_on_road(self.map, x=ego_data['location'][0],
                                     y=ego_data['location'][1],
                                     z=ego_data['location'][2]) and \
           not ego_data['is_in_junction']:
            answer = "No, because the ego vehicle is not in any lane, it went out of the road."
            object_tags = []
            change_flag = False
            change_dir = 4

        self.add_qas_questions(qa_list=qas_conversation_ego,
                                qid=12,
                                chain=3,
                                layer=8,
                                qa_type='planning',
                                connection_up=[(6, 0)] ,
                                connection_down=[(3, 9)],
                                question=question,
                                answer=answer,
                                object_tags=object_tags)

        return change_flag, change_dir, answer, object_tags, change_lane_on_highway, special_change_lane_reason, ood_flag


    qas_conversation_ego = []
    
    res = determine_whether_ego_needs_to_change_lanes_due_to_obstruction(qas_conversation_ego,
                                                                    scenario_name,
                                                                    vehicles_by_id,
                                                                    static_objects,
                                                                    measurements,
                                                                    ego_data, important_objects, key_object_infos)

    obstacle_change_flag, obstacle_change_dir, plan_ans1, obs_ans, obj_tags1, must_stop_str, must_brake_str = res

    res = determine_whether_ego_needs_to_change_lanes_due_to_other_factor(qas_conversation_ego,
                                                                    scenario_name,
                                                                    vehicles_by_id,
                                                                    static_objects,
                                                                    measurements,
                                                                    ego_data, important_objects, key_object_infos)

    other_change_flag, other_change_dir, plan_ans2, obj_tags2, change_lane_on_highway, extra_changing_reason, ood_flag = res

    print_debug(f"[debug] all flags before analysing lane_change: obstacle_change_flag = {obstacle_change_flag}, obstacle_change_dir = {obstacle_change_dir}, must_stop_str = {must_stop_str}, must_brake_str = {must_brake_str}, other_change_flag = {other_change_flag}, other_change_dir = {other_change_dir}, change_lane_on_highway = {change_lane_on_highway}, extra_changing_reason = {extra_changing_reason}, ood_flag = {ood_flag}")

    question = "Must the ego vehicle change lane or deviate from the lane now? why?"
    answer = "No, the ego vehicle can stay on its current lane."

    scenario_name = scenario_name.split('_')[0]
    still_changing_lane_flag = False
    if (obstacle_change_flag or other_change_flag or self.opposite_flag or ood_flag):
        answer = ""
        if obstacle_change_flag:
            if scenario_name in ['ParkingExit']: # exit the parking space
                answer = plan_ans1
            else:
                answer = obs_ans
                if plan_ans1.startswith("Yes, "):
                    plan_ans1 = plan_ans1[len("Yes, "):]
                if plan_ans1.startswith("No, "):
                    plan_ans1 = plan_ans1[len("No, "):]
                if "still changing lane" in plan_ans1:
                    answer = f"The ego vehicle should align to the target lane beacuse {plan_ans1[0].lower()}{plan_ans1[1:]}"
                    still_changing_lane_flag = True
                else:
                    answer = f"{answer} So {plan_ans1[0].lower()}{plan_ans1[1:]}"
        if other_change_flag or self.opposite_flag or ood_flag: # when the ego vehicle is in the opposite lane, must change back ASAP
            if answer != "":
                answer = f"{answer} Secondly,"
                if plan_ans2.startswith("Yes, "):
                    plan_ans2 = plan_ans2[len("Yes, "):]
                if plan_ans2.startswith("No, "):
                    plan_ans2 = plan_ans2[len("No, "):]
            if "still changing lane" in plan_ans2:
                still_changing_lane_flag = True
            answer = f"{answer} {plan_ans2[0].upper()}{plan_ans2[1:]}"

    # print_debug(f"[debug] frame {self.current_measurement_index}, obstacle_change_flag = {obstacle_change_flag}, other_change_flag = {other_change_flag}, opposite_flag = {self.opposite_flag}, ood_flag = {ood_flag}")

    obj_tags1.extend(obj_tags2)
    
    # old question add entrance was here
    
    # add vehicles in rear on target lane into key objects\
    # and consider crossing as well
    x = (measurements['x_command_far'] - measurements['x'])**2
    y = (measurements['y_command_far'] - measurements['y'])**2
    command_distance = np.sqrt(x + y)
    final_change_dir = 4
    if measurements['command_near'] == 5 or (measurements['command_far'] == 5 and command_distance < 15.0): 
        if ego_data['lane_change'] in [2, 3]: # left
            final_change_dir = 2
    if measurements['command_near'] == 6 or (measurements['command_far'] == 6 and command_distance < 15.0): 
        if ego_data['lane_change'] in [1, 3]: # right
            final_change_dir = 1
    
    if obstacle_change_flag: final_change_dir = obstacle_change_dir
    if other_change_flag or ood_flag:
        if obstacle_change_flag:
            final_change_dir = min(obstacle_change_dir, other_change_dir)
        else:
            final_change_dir = other_change_dir

    # print_debug(f"[debug] after calculation, final_change_dir = {final_change_dir}")
    
    look_back = True
    lane_obstacle_list = []
    # ahead_clear_distance = get_clear_distance_of_lane(vehicles_by_id.values(), 0, False)

    target_lane_occupied = False
    target_lane_back_clear = True

    NORMAL_OFFSET = LANE_CHANGE_NORMAL_OFFSET
    OPPOSITE_FRONT_OFFSET = LANE_CHANGE_OPPOSITE_FRONT_OFFSET

    FRONT_DANGER_DISTANCE = LANE_CHANGE_FRONT_DANGER_DISTANCE
    BACK_DANGER_DISTANCE = LANE_CHANGE_BACK_DANGER_DISTANCE
    # if change_lane_on_highway or 'LaneChange' in scenario_name:
    #     BACK_DANGER_DISTANCE = LANE_CHANGE_HIGHWAY_BACK_DANGER_DISTANCE
    if self.in_carla:
        danger_interval = LANE_CHANGE_DANGER_INTERVAL
        if scenario_name in ['YieldToEmergencyVehicle']:
            danger_interval = EMERGENCY_VEHICLE_DANGER_INTERVAL
        BACK_DANGER_DISTANCE = max(LANE_CHANGE_BACK_DANGER_DISTANCE, self.ideal_flow_speed * danger_interval)
    if scenario_name in ['ParkingExit'] and (not self.in_carla):
        BACK_DANGER_DISTANCE = LANE_CHANGE_PARKING_EXIT_BACK_DANGER_DISTANCE
    OPPOSITE_DANGER_DISTANCE = LANE_CHANGE_OPPOSITE_DANGER_DISTANCE

    if scenario_name in self.circumvent_scenarios and \
       self.distance_to_circumvent_obstacle is not None and \
       max(CHANGE_LANE_THRESHOLD, ego_data['speed'] * BRAKE_INTERVAL) >= self.distance_to_circumvent_obstacle >= 0:
        OPPOSITE_FRONT_OFFSET += self.distance_to_circumvent_obstacle + self.get_obstacle_pass_offset(scenario_name)
        OPPOSITE_DANGER_DISTANCE += self.distance_to_circumvent_obstacle + self.get_obstacle_pass_offset(scenario_name)
    lane_change_back_clear_threshold = CARLA_LANE_CHANGE_BACK_CLEAR_THRESHOLD if self.in_carla else LANE_CHANGE_BACK_CLEAR_THRESHOLD

    print_debug(f"[debug] OPPOSITE_FRONT_OFFSET = {OPPOSITE_FRONT_OFFSET}, OPPOSITE_DANGER_DISTANCE = {OPPOSITE_DANGER_DISTANCE}")

    if self.opposite_flag == False:
        if final_change_dir in [2, 3]: # left
            # consider left lane
            if ego_data['lane_change'] in [0, 1]:
                # change to opposite
                look_back = False
                target_lane_back_clear = False # should now follow near vehicle on the opposite lane
            offset = NORMAL_OFFSET if look_back else OPPOSITE_FRONT_OFFSET # opposite lane is more dangerous
            front_danger_distance = FRONT_DANGER_DISTANCE if look_back else OPPOSITE_DANGER_DISTANCE
            tmp_list = get_vehicle_in_lane_within_threshold(vehicles_by_id.values(), -1,
                                                            self.lane_clear_threshold + offset,
                                                            backwards=look_back)
            # always check forward
            if look_back is True:
                rev_list = get_vehicle_in_lane_within_threshold(vehicles_by_id.values(), -1,
                                                                self.lane_forward_threshold,
                                                                backwards=False)
                tmp_list.extend(rev_list)
            
            if scenario_name in self.highway_change_lane_scenarios or scenario_name == 'InterurbanAdvancedActorFlow':
                for vehicle in vehicles_by_id.values():
                    if vehicle['position'][1] < 0.0 and vehicle['position'][0] > LANE_CHANGE_REAR_ALERT_THRESHOLD and \
                        vehicle['distance'] < LANE_CHANGE_DISTANCE_ALERT_THRESHOLD:
                        existed = False
                        for exist_vehicle in tmp_list:
                            if exist_vehicle['id'] == vehicle['id']:
                                existed = True
                        if not existed:
                            tmp_list.append(vehicle)

            if tmp_list:
                for x in tmp_list:
                    if ((x['position'][0] > 0.0 and x['distance'] < front_danger_distance) or \
                        (x['position'][0] <= 0.0 and x['distance'] < BACK_DANGER_DISTANCE)) and \
                        (not (x['position'][0] <= -LANE_CHANGE_STATIC_IGNORE_DISTANCE and x['speed'] < STOP_VEHICLE_SPEED)):
                        target_lane_occupied = True
                        if x['position'][0] <= lane_change_back_clear_threshold:
                            target_lane_back_clear = False
                        # print_debug(f"[debug] vehicle {x['id']} x={x['position'][0]}") #
                    dir_str = "left" if look_back else "opposite"
                    lane_obstacle_list.append([dir_str, x])
        
        if final_change_dir in [1, 3]: # right
            # consider right lane
            offset = NORMAL_OFFSET if look_back else OPPOSITE_FRONT_OFFSET
            front_danger_distance = FRONT_DANGER_DISTANCE if look_back else OPPOSITE_DANGER_DISTANCE
            tmp_list = get_vehicle_in_lane_within_threshold(vehicles_by_id.values(), 1,
                                                            self.lane_clear_threshold + offset,
                                                            backwards=look_back)
            # always check forward
            if look_back is True:
                rev_list = get_vehicle_in_lane_within_threshold(vehicles_by_id.values(), 1,
                                                                self.lane_forward_threshold,
                                                                backwards=False)
                tmp_list.extend(rev_list)
            
            if scenario_name in self.highway_change_lane_scenarios or scenario_name == 'InterurbanAdvancedActorFlow':
                for vehicle in vehicles_by_id.values():
                    if vehicle['position'][1] > 0.0 and vehicle['position'][0] > LANE_CHANGE_REAR_ALERT_THRESHOLD and \
                        vehicle['distance'] < LANE_CHANGE_DISTANCE_ALERT_THRESHOLD:
                        existed = False
                        for exist_vehicle in tmp_list:
                            if exist_vehicle['id'] == vehicle['id']:
                                existed = True
                        if not existed:
                            tmp_list.append(vehicle)

            if tmp_list:
                for x in tmp_list:
                    # if both directions, consider left
                    if final_change_dir not in [3] and scenario_name not in self.circumvent_scenarios and \
                        ((x['position'][0] > 0.0 and x['distance'] < front_danger_distance) or \
                            (x['position'][0] <= 0.0 and x['distance'] < BACK_DANGER_DISTANCE)) and \
                        (not (x['position'][0] <= -LANE_CHANGE_STATIC_IGNORE_DISTANCE and x['speed'] < STOP_VEHICLE_SPEED)):
                        target_lane_occupied = True
                        if x['position'][0] <= lane_change_back_clear_threshold:
                            target_lane_back_clear = False 
                        # print_debug(f"[debug] vehicle {x['id']} x={x['position'][0]}") #
                    lane_obstacle_list.append(["right", x])
    else:
        # opposite situation
        shoulder_vehicle = [x for x in vehicles_by_id.values() if x['lane_type_str'] == 'Shoulder']
        shoulder_max_position_x = -INF_MAX
        if isinstance(shoulder_vehicle, list) and len(shoulder_vehicle) > 0:
            shoulder_max_position_x = max(vehicle['position'][0] for vehicle in shoulder_vehicle)
        look_back = True
        offset = NORMAL_OFFSET if look_back else OPPOSITE_FRONT_OFFSET # opposite lane is more dangerous
        tmp_list = get_vehicle_in_lane_within_threshold(vehicles_by_id.values(), -1,
                                                        self.lane_clear_threshold + offset,
                                                        backwards=look_back)
        # always check forward
        if look_back is True:
            rev_list = get_vehicle_in_lane_within_threshold(vehicles_by_id.values(), -1,
                                                            self.lane_forward_threshold,
                                                            backwards=False)
            tmp_list.extend(rev_list)

        if tmp_list:
            for x in tmp_list:
                if x['position'][0] > shoulder_max_position_x:
                    lane_obstacle_list.append(["right", x])
    
    # print_debug(f"[debug] target_lane_occupied = {target_lane_occupied}, target_lane_back_clear = {target_lane_back_clear}")
    # left_must_wait_flag = False
    # right_must_wait_flag = False
    # MUST_STOP_THRESHOLD = 8.0
    
    if lane_obstacle_list:
        for pair in lane_obstacle_list:
            
            relevant_obj = pair[1]
            direction_str = pair[0]
            # if relevant_obj['distance'] < MUST_STOP_THRESHOLD:
            #     if direction_str == 'left':
            #         left_must_wait_flag = True
            #     if direction_str == 'right':
            #         right_must_wait_flag = True

            # print_debug(f"[debug] frame = {self.current_measurement_index}, obstacle_vehicle {relevant_obj}")
            if abs(relevant_obj['speed']) < STOP_VEHICLE_SPEED:
                # static cars are obstacles
                continue

            rough_pos_str = get_pos_str(relevant_obj['position'])
            color_str = ""
            if relevant_obj.get('color') is not None:
                color_str = rgb_to_color_name(relevant_obj['color']) + ' '
                if relevant_obj['color'] == [0, 28, 0] or relevant_obj['color'] == [12, 42, 12]:
                    color_str = 'dark green '
                elif relevant_obj['color'] == [211, 142, 0]:
                    color_str = 'yellow '
                elif relevant_obj['color'] == [145, 255, 181]:
                    color_str = 'blue '
                elif relevant_obj['color'] == [215, 88, 0]:
                    color_str = 'orange '

            category = "Vehicle"
            cross_reason = f"is in the {pair[0]} lane which ego vehicle wants to change onto"
            if (relevant_obj['position'][0] <= -LANE_CHANGE_STATIC_IGNORE_DISTANCE and relevant_obj['speed'] < STOP_VEHICLE_SPEED):
                continue # ignored static vehicles at back
            if relevant_obj['position'][0] > LANE_CHANGE_FRONT_BACK_DIVIDER:
                if look_back == False or (relevant_obj['distance'] < FRONT_DANGER_DISTANCE and (not target_lane_back_clear)):
                    cross_action = f"changes to the {pair[0]} lane immediately"
                else:
                    cross_action = f"drives too fast when changing to the {pair[0]} lane"
            else:
                if relevant_obj['distance'] < BACK_DANGER_DISTANCE:
                    cross_action = f"changes to the {pair[0]} lane immediately"
                else:
                    cross_action = f"changes to the {pair[0]} lane but drives too slowly"

            if scenario_name in ['InvadingTurn']:
                cross_reason = f"invades the ego vehicle's lane from the opposite direction"
                cross_action = f"stay at the center of current lane without shifting aside"
            visual_description = f"{color_str}{relevant_obj['base_type']}"
            important_object_str = f"the {color_str}{relevant_obj['base_type']} {rough_pos_str}"
            
            consider_vehicle([relevant_obj])
            set_vehicle_overlap(relevant_obj['id'], cross_reason, cross_action)
            
            if not is_object_in_key_object(key_object_infos, relevant_obj):
                important_objects.append(important_object_str)
                # projected_points, projected_points_meters = project_all_corners(relevant_obj, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                # Generate a unique key and value for the vehicle object
                project_dict = get_project_camera_and_corners(relevant_obj, self.CAM_DICT)
                key, value = self.generate_object_key_value(
                    id=relevant_obj['id'],
                    category=category,
                    visual_description=visual_description,
                    detailed_description=important_object_str,
                    object_count=len(key_object_infos),
                    is_role=False,
                    is_dangerous=True,
                    obj_dict=relevant_obj,
                    projected_dict=project_dict
                )
                key_object_infos[key] = value
            else:
                for key, key_dict in key_object_infos.items():
                    if key_dict['id'] == relevant_obj['id']:
                        key_dict['is_dangerous'] = True
    
    # special_cases
    special_case_vehicles = []
    special_reasons = []
    special_actions = []

    highway_merging_danger_flag = False

    # fixed in some scenario, near important object neglected
    for vehicle in vehicles_by_id.values():
        angle = math.degrees(math.atan(vehicle['position'][1] / vehicle['position'][0]))
        # print_debug(f"[debug] vehicle_id = {vehicle['id']}, position = {vehicle['position']}, scenario_name in self.enter_highway_scenarios = {scenario_name in self.enter_highway_scenarios}, vehicle['junction_id'] == ego_data['junction_id'] = {vehicle['junction_id'] == ego_data['junction_id']}, vehicle['is_in_junction'] and ego_data['is_in_junction'] = {vehicle['is_in_junction'] and ego_data['is_in_junction']}, vehicle['distance'] < HIGHWAY_MEET_MAX_DISTANCE = {vehicle['distance'] < HIGHWAY_MEET_MAX_DISTANCE}, vehicle['position'][0] > -HIGHWAY_MEET_MIN_X = {vehicle['position'][0] > HIGHWAY_MEET_MIN_X}, vehicle['id'] not in self.vehicle_ids_following_ego = {vehicle['id'] not in self.vehicle_ids_following_ego}, (vehicle['position'][1] < 0.0 and vehicle['yaw'] > HIGHWAY_MEET_MIN_ANGLE) = {(vehicle['position'][1] < 0.0 and vehicle['yaw'] > HIGHWAY_MEET_MIN_ANGLE)}, (vehicle['position'][1] > 0.0 and vehicle['yaw'] < -HIGHWAY_MEET_MIN_ANGLE) = {(vehicle['position'][1] > 0.0 and vehicle['yaw'] < -HIGHWAY_MEET_MIN_ANGLE)}")

        if self.opposite_flag == True and \
           ego_data['num_lanes_opposite_direction'] <= 1 and \
           vehicle['lane_type_str'] == 'Shoulder' and \
           vehicle['position'][0] >= OPPOSITE_SHOULDER_BLOCKER_MIN_X:
            # print(f"{scenario_name}: {self.current_measurement_index}: {vehicle['id']} blocks original lane!")
            relevant_obj = vehicle
            cross_reason = f"occupies the original lane that the ego vehicle urgently needs to return to"
            cross_action = f"changes back to original lane on the right immediately"
            special_case_vehicles.append(relevant_obj)
            special_reasons.append(cross_reason)
            special_actions.append(cross_action)

        elif scenario_name in self.enter_highway_scenarios and vehicle['junction_id'] == ego_data['junction_id'] and \
                vehicle['is_in_junction'] and ego_data['is_in_junction'] and \
                vehicle['distance'] < HIGHWAY_MEET_MAX_DISTANCE and vehicle['position'][0] > HIGHWAY_MEET_MIN_X and \
                abs(vehicle['position'][1]) < HIGHWAY_MEET_MAX_Y and \
                vehicle['approaching_dot_product'] < 0.0 and \
                vehicle['id'] not in self.vehicle_ids_following_ego and \
                ((vehicle['position'][1] < 0.0 and vehicle['yaw'] > HIGHWAY_MEET_MIN_ANGLE) or 
                (vehicle['position'][1] > 0.0 and vehicle['yaw'] < -HIGHWAY_MEET_MIN_ANGLE)):
            print_debug(f"[debug] vehicle id={vehicle['id']} meets the ego vehicle!")
            self.front_merge_vehicle_ids[vehicle['id']] = vehicle['distance']
            relevant_obj = vehicle
            verb = "meets"
            cross_action = "merges into the target traffic flow too fast without yielding"
            if vehicle['position'][0] < HIGHWAY_MEET_FRONT_REAR_DIVIDER:
                verb = "might meet"
                cross_action = "either merges into the target fast-moving traffic flow too slowly or does not yield"
                self.merging_and_needs_accelerate = True
            else:
                highway_merging_danger_flag = True
            if HIGHWAY_MEET_STOP_MIN_X < vehicle['position'][0] < HIGHWAY_MEET_STOP_MAX_X and \
               abs(vehicle['position'][1]) < HIGHWAY_MEET_STOP_MAX_Y and \
               (abs(vehicle['position'][1]) > HIGHWAY_MEET_STOP_MIN_Y or vehicle['position'][0] > BACK_CONSIDER_THRESHOLD) and \
               vehicle['speed'] > max(SLOW_VEHICLE_SPEED, ego_data['speed'] * SLOW_VEHICLE_RATIO):
                print_debug(f"[debug] vehicle id={vehicle['id']} is a must-stop reason for merging")
                verb = "meets"
                cross_action = "merges into the target traffic flow without yielding"
                self.merging_and_needs_stop = True
                self.merging_danger_vehicles.append(vehicle)
            cross_reason = f"{verb} the ego vehicle in the highway merging area"
            special_case_vehicles.append(relevant_obj)
            special_reasons.append(cross_reason)
            special_actions.append(cross_action)
        
        elif 'Cut' in scenario_name and vehicle.get('vehicle_cuts_in', False) == True and \
            vehicle['distance'] < CUT_IN_CONSIDER_DISTANCE and vehicle['position'][0] > BACK_CONSIDER_THRESHOLD:
            cross_reason = f"is cutting into the ego vehicle's lane"
            cross_action = f"keeps driving forward without decelerating"
            if vehicle['speed'] >= ego_data['speed'] * SLOW_VEHICLE_RATIO and vehicle['distance'] > CUT_IN_STOP_DISTANCE:
                cross_action = f"drives forward too fast"
            special_case_vehicles.append(vehicle)
            special_reasons.append(cross_reason)
            special_actions.append(cross_action)

        # elif vehicle['distance'] <= 10.0 and abs(angle) <= 15.0 and is_vehicle_in_camera(self.CAMERA_FRONT, vehicle):
        #     # print(f"{scenario_name}: {self.current_measurement_index}: {vehicle['id']} is too close right in front!")
        #     relevant_obj = vehicle
        #     cross_reason = f"is too close and right in front of the ego vehicle"
        #     special_case_vehicles.append(relevant_obj)
        #     special_reasons.append(cross_reason)
        #     # print(f"{scenario_name}: {self.current_measurement_index}: {vehicle['id']}: considered = {is_vehicle_considered(relevant_obj)}, is_found = {is_found_in_key}")    
    
    
    for relevant_obj, cross_reason, cross_action in zip(special_case_vehicles, special_reasons, special_actions):
        is_found_in_key = False
        for key, key_dict in key_object_infos.items():
            if key_dict['id'] == relevant_obj['id']:
                is_found_in_key = True
                key_dict['is_dangerous'] = True
        consider_vehicle([relevant_obj])
        set_vehicle_overlap(relevant_obj['id'], cross_reason, cross_action)
        if not is_found_in_key:
            rough_pos_str = get_pos_str(relevant_obj['position'])
            color_str = ""
            if relevant_obj.get('color') is not None:
                color_str = rgb_to_color_name(relevant_obj['color']) + ' '
                if relevant_obj['color'] == [0, 28, 0] or relevant_obj['color'] == [12, 42, 12]:
                    color_str = 'dark green '
                elif relevant_obj['color'] == [211, 142, 0]:
                    color_str = 'yellow '
                elif relevant_obj['color'] == [145, 255, 181]:
                    color_str = 'blue '
                elif relevant_obj['color'] == [215, 88, 0]:
                    color_str = 'orange '

            category = "Vehicle"
            visual_description = f"{color_str}{relevant_obj['base_type']}"
            important_object_str = f"the {color_str}{relevant_obj['base_type']} {rough_pos_str}"
            
            del_object_in_key_info(key_object_infos, [relevant_obj])
            
            # old_str, _, _ = get_vehicle_str(relevant_obj)
            important_objects.append(important_object_str)
            # projected_points, projected_points_meters = project_all_corners(relevant_obj, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
            # Generate a unique key and value for the vehicle object
            project_dict = get_project_camera_and_corners(relevant_obj, self.CAM_DICT)
            key, value = self.generate_object_key_value(
                id=relevant_obj['id'],
                category=category,
                visual_description=visual_description,
                detailed_description=important_object_str,
                object_count=len(key_object_infos),
                is_role=False,
                is_dangerous=True,
                obj_dict=relevant_obj,
                projected_dict=project_dict
            )
            key_object_infos[key] = value
    
    stop_template_prefix = "The ego vehicle needs to stop and wait for a chance to change lane, " +\
                        "as our lane is blocked, and there exists a vehicle too close to us in the "
    stop_template_subfix = " lane we want to enter."

    drive_template_prefix = "The ego vehicle needs to drive slowly in current lane and wait for a chance to change lane, " +\
                        "as there exists a vehicle too close to us in the "
    drive_template_subfix = " lane we want to enter."

    must_wait_for_lane_change_str = None

    
    ###### re-correction of lane change question ######

    # print_debug(f"[debug] before re-correction, final_change_dir = {final_change_dir}")

    lane_change_dir = final_change_dir
    check_back = False if ego_data['lane_change'] in [0] else True
    lane_occupied_flag = target_lane_occupied #  and check_back
    target_lane_back_clear = target_lane_back_clear and check_back
    lane_change_flag = lane_change_dir in [1, 2, 3]
    obstacle_obj_tag=obj_tags1
    occupied_str = 'is occupied' if lane_occupied_flag else 'is busy'
    
    final_lane_change_flag = lane_change_flag
    # if (change_lane_on_highway and scenario_name in self.highway_change_lane_scenarios) or \
    #     (lane_change_flag and scenario_name in self.circumvent_scenarios) or \
    #     (lane_change_flag and scenario_name in ['ParkingExit']) or \
    #     (other_change_flag and (scenario_name in ['LaneChange', 'YieldToEmergencyVehicle'] or 'LaneChange' in scenario_name)):
    if final_lane_change_flag and \
       not ("deviate slightly" in plan_ans1 and 'InvadingTurn' in scenario_name):
        # print_debug(f"[debug] index = {self.current_measurement_index}, lane_occupied = {target_lane_occupied}, clear = {target_lane_back_clear}")
        if lane_occupied_flag and (not target_lane_back_clear):
            answer = answer + f" But not now because the target lane {occupied_str}."
            final_lane_change_flag = False
    
    if not self.in_carla:
        gt_res = detect_future_lane_change_by_time(self.map, ego_data['id'], self.current_measurement_path, k=25)
    else:
        original_lane_id = ego_data['lane_id']
        new_lane_id = ego_data['lane_id']
        gt_res = original_lane_id, new_lane_id, False
    original_lane_id, new_lane_id, changed_for_real = gt_res

    self.answer43_changelane = answer

    # print_debug(f"[debug] after re-correction, final_change_dir = {final_change_dir}, final_lane_change_flag = {final_lane_change_flag}")

    if not ('ParkingExit' in scenario_name and ego_data['lane_type_str'] == 'Parking') and \
       not is_on_road(self.map, x=ego_data['location'][0],
                                 y=ego_data['location'][1],
                                 z=ego_data['location'][2]) and \
       not ego_data['is_in_junction']:
        answer = "No, because the ego vehicle is not in any lane, it went out of the road."
        object_tags = []
        final_change_dir = 0
        final_lane_change_flag = False

    # print_debug(f"[debug] finally, final_change_dir = {final_change_dir}, final_lane_change_flag = {final_lane_change_flag}")

    self.add_qas_questions(qa_list=qas_conversation_ego,
                            qid=13,
                            chain=1, 
                            layer=1, 
                            qa_type='planning',
                            connection_up=[(6,0)], 
                            connection_down=[(1,0)], 
                            question=question,
                            answer=answer,
                            object_tags=obj_tags1)

    ego_speed = ego_vehicle_data['speed']

    actor_stop = [traffic_light_info]
    object_tags = [traffic_light_object_tags]
    actor_names = ['traffic light']
    for key, value in self.traffic_sign_map.items():
        actor_stop.append(traffic_sign_info[key] if len(traffic_sign_info[key]) > 0 else None)
        object_tags.append(traffic_sign_object_tags[key])
        actor_names.append(key.replace("_", " "))
    
    sign_behavior_list = []
    already_stopped_at_stop_sign = False
    for actor, actor_name, tags in zip(actor_stop, actor_names, object_tags):
        if actor_name == 'speed limit' and self.current_speed_limit is not None and\
        self.future_speed_limit is None:
            question = f"What should the ego vehicle do based on the passed speed limit sign?"
            answer = f"The ego vehicle's speed should not exceed {self.current_speed_limit} km/h."
            self.add_qas_questions(qa_list=qas_conversation_ego, 
                                qid=9,
                                chain=1, 
                                layer=1, 
                                qa_type='planning',
                                connection_up=[(6,0)], 
                                connection_down=[(1,0)], 
                                question=question,
                                answer=answer,
                                object_tags=[])
            sign_name = "speed limit sign passed"
            sign_behaviour = f"should not exceed the speed of {self.current_speed_limit} km/h"
            sign_behavior_list.append([sign_name, sign_behaviour, []])
        elif not (actor_name == 'traffic light' and ego_data['is_in_junction']):
            res = determine_ego_action_based_on_actor(actor,
                                                      actor_name,
                                                      ego_speed,
                                                      ego_data,
                                                      qas_conversation_ego, 
                                                      stop_signs,
                                                      tags)
            sign_name, sign_behaviour, already_stopped_at_this = res
            already_stopped_at_stop_sign = already_stopped_at_stop_sign or already_stopped_at_this
            self.stopped_at_stop_sign = already_stopped_at_stop_sign or self.stopped_at_stop_sign
            sign_behavior_list.append([sign_name, sign_behaviour, tags])

    res = determine_braking_requirement(qas_conversation_ego,
                                    pedestrians,
                                    measurements,
                                    scene_data,
                                    vehicles_by_id,
                                    ego_data,
                                    scenario_name,
                                    traffic_light_info,
                                    traffic_sign_info['stop_sign'],
                                    static_objects,
                                    target_lane_occupied=target_lane_occupied,
                                    target_lane_back_clear=target_lane_back_clear,
                                    lane_change_dir=final_change_dir,
                                    still_changing_lane_flag=still_changing_lane_flag,
                                    highway_merging_danger_flag=highway_merging_danger_flag,
                                    already_stopped_at_stop_sign=already_stopped_at_stop_sign,
                                    must_stop_str=must_stop_str,
                                    must_brake_str=must_brake_str,
                                    obstacle_obj_tag=obj_tags1,
                                    change_lane_on_highway=change_lane_on_highway,
                                    extra_changing_reason=extra_changing_reason)
    
    final_brake_flag, final_stop_flag, suggest_stop_flag = res
    if lane_change_flag == True and final_stop_flag == True:
        final_lane_change_flag = True # stop when attempting to change lane
    
    all_tags = [x for x in object_tags if x is not None]
    object_tags = []
    for tag_list in all_tags:
        object_tags.extend(tag_list)
    sign_list_str = ""
    answer = ""
    answer2 = ""
    for sign in sign_behavior_list:
        if sign[0] is not None:
            if sign_list_str == "":
                sign_list_str = f"the {sign[0]}({sign[2]})"
            else:
                sign_list_str = f"{sign_list_str}, the {sign[0]}({sign[2]})"
            answer = f"{answer} Based on the {sign[0]}({sign[2]}), the ego vehicle {sign[1]}."
            answer2 = f"{answer2} The {sign[0]}({sign[2]}). Based on it, the ego vehicle {sign[1]}."

    if sign_list_str != "":
        question = f"The list of traffic lights and signs affecting the ego vehicle in current scene is: {sign_list_str}. " +\
                    "Based on these traffic signs, what actions should the ego vehicle take respectively?"
        
        self.add_qas_questions(qa_list=qas_conversation_ego,
                                    qid=14,
                                    chain=1, 
                                    layer=1, 
                                    qa_type='planning',
                                    connection_up=[(6,0)], 
                                    connection_down=[(1,0)], 
                                    question=question,
                                    answer=answer,
                                    object_tags=object_tags)

    question = f"Identify all traffic lights and signs affecting the ego vehicle in current scene. " +\
                "Based on these traffic signs, what actions should the ego vehicle take respectively?"

    if sign_list_str == "":
        answer2 = "There's no traffic light or sign affecting the ego vehicle right now."
    
    self.add_qas_questions(qa_list=qas_conversation_ego,
                                qid=15, 
                                chain=1, 
                                layer=1, 
                                qa_type='planning',
                                connection_up=[(6,0)], 
                                connection_down=[(1,0)], 
                                question=question,
                                answer=answer2,
                                object_tags=object_tags)

    add_speed_limit_question(qas_conversation_ego,
                                measurements)
    
    if obstacle_change_flag and final_lane_change_flag:
        self.last_left_lane = ego_data['lane_id']
        self.last_special_move_index = self.current_measurement_index
    
    print_debug(f"[debug] frame {self.current_measurement_index}, final_change_dir = {final_change_dir}, final_lane_change_flag = {final_lane_change_flag}, final_brake_flag = {final_brake_flag}, final_stop_flag = {final_stop_flag}")

    return qas_conversation_ego, important_objects, key_object_infos, \
           final_change_dir, final_lane_change_flag, changed_for_real, \
           final_brake_flag, final_stop_flag