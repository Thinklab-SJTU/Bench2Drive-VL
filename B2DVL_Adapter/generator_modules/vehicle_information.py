import numpy as np
import carla
from .configs import *
from .offline_map_calculations import *
from .hyper_params import *

def generate_vehicle_information(self, other_vehicles, ego_vehicle, important_objects, key_object_infos, 
                                            scene_data, vehicles_by_id, current_measurement, scenario):
    """
    Generates information and question-answer pairs for vehicles in the scene.

    Answers the questions:
    - Where on the road is the vehicle located?
    - Where is the vehicle going?
    - What is the moving status of the vehicle?
    - The ego vehicle {command_str}. Is {vehicle_location_description} potentially crossing the 
        path of the ego vehicle?

    Args:
        other_vehicles (list): List of dictionaries containing information about other vehicles in the scene.
        ego_vehicle (dict): Dictionary containing information about the ego vehicle.
        important_objects (list): List to store important objects in the scene.
        key_object_infos (dict): Dictionary to store information about objects in the scene.
        num_lanes_ego (int): Number of lanes for the ego vehicle.
        vehicles_by_id (dict): Dictionary mapping vehicle IDs to vehicle information.
        current_measurement (dict): Dictionary containing current measurement data.
        scenario (str): The current scenario.

    Returns:
        qas_conversation_vehicle (list): List of question-answer pairs related to vehicles.
        important_objects (list): Updated list of important objects in the scene.
        key_object_infos (dict): Updated dictionary of object information.
    """

    def determine_path_crossing(current_measurement, ego_vehicle, other_vehicle_location_description, 
                        other_vehicle, vehicles_by_id, other_vehicle_description, scenario, 
                        ego_distance_to_junction, other_vehicle_points_towards_ego, 
                        other_vehicle_heading_angle_deg, pointing_towards_junction, is_ego_on_highway, 
                        is_ego_in_accel_lane, is_other_veh_in_accel_lane):
        """
        Answers: "The ego vehicle {command_str}. Is {vehicle_location_description} potentially crossing the 
            path of the ego vehicle?"
        
        Args:
            current_measurement (dict): Current measurement data.
            ego_vehicle (dict): Information about the ego vehicle.
            other_vehicle_location_description (str): Description of the other vehicle's location.
            other_vehicle (dict): Information about the other vehicle.
            vehicles_by_id (dict): Dictionary of vehicles by their ID.
            other_vehicle_description (str): Description of the other vehicle.
            scenario (str): The current scenario.
            ego_distance_to_junction (float): Distance of the ego vehicle to the next junction.
            other_vehicle_points_towards_ego (bool): True if the other vehicle is pointing towards the ego vehicle.
            other_vehicle_heading_angle_deg (float): Heading angle of the other vehicle in degrees.
            pointing_towards_junction (bool): True if the other vehicle is pointing towards the junction.
            is_ego_on_highway (bool): True if the ego vehicle is on a highway.
            is_ego_in_accel_lane (bool): True if the ego vehicle is in an acceleration lane.
            is_other_veh_in_accel_lane (bool): True if the other vehicle is in an acceleration lane.
            qas_conversation_vehicle (list): List of question-answer pairs for the vehicle.

        Returns:
            crossed (bool): True if path crosses
            command_str (str): Command string of ego vehicle
            reason (str): Reason why path crosses
            action (str): Action may lead to collision
        """            

        scenario = scenario.split('_')[0]

        # Map command integers to their corresponding string descriptions
        command_int = get_command_int_by_current_measurement(current_measurement=current_measurement,
                                                             ego_vehicle=ego_vehicle)

        # the next intersection -> the intersection
        command_map = {
            1: 'turns left at the intersection',
            2: 'turns right at the intersection',
            3: 'drives straight at the intersection',
            4: 'follows the road',
            # 5: f'does a lane change to the left in {int(command_distance)} m',
            # 6: f'does a lane change to the right in {int(command_distance)} m',
            5: f'does a lane change to the left',
            6: f'does a lane change to the right',
        }
        command_str = command_map[command_int]

        # Update command string if the ego vehicle is already in a junction
        if ego_vehicle['is_in_junction'] or ego_vehicle.get('distance_to_junction', INF_MAX) < JUNCTION_EXTEND_DISTANCE:
            command_str = command_str.replace('turns', 'continues turning').replace('drives', 'continues driving')
            command_str = command_str.replace('next intersection', 'current intersection')

        theta = calculate_angle_between_two_vehicles(ego_vehicle, other_vehicle)

        question = f"The ego vehicle {command_str}. Is {other_vehicle_location_description} potentially " \
                        "crossing the path of the ego vehicle? If so, why?"
        actor_in_front = None
        same_future_road = any([x==y for x in other_vehicle['next_road_ids'] for y in ego_vehicle['next_road_ids']])
        crossed = False
        reason = ""
        action = ""

        basic_intersect_flag = False

        if ego_vehicle['is_in_junction'] and command_int not in [1, 2, 3]:
            if ego_vehicle['steer'] < -TURN_STEER_THRESHOLD: command_int = 1
            elif ego_vehicle['steer'] > TURN_STEER_THRESHOLD: command_int = 2
            
        if scenario in self.merging_scenarios and command_int in [1, 2, 3] and \
           ego_vehicle['distance_to_junction'] < JUNCTION_EXTEND_DISTANCE:
            # Basic intersection check    
            res = determine_basic_vector_crossing(command_int=command_int,
                                                other_vehicle=other_vehicle,
                                                whitelist=self.accelerate_white_list,
                                                following_list=self.vehicle_ids_following_ego,
                                                first_appear_history=self.first_appear_intervals)
            basic_intersect_flag, basic_reason, basic_action, _, _ = res
            if other_vehicle['id'] in self.accelerate_black_list and \
            other_vehicle['approaching_dot_product'] < 0 and \
            other_vehicle['distance'] < self.accelerate_black_list[other_vehicle['id']]:
                basic_intersect_flag = True
                if other_vehicle['id'] in self.accelerate_white_list:
                    basic_reason = "will cross the ego vehicle's path in the future"
                    basic_action = "continues driving slowly"
                else:
                    basic_reason = "is crossing the ego vehicle's path and is still approaching"
                    basic_action = "continues driving without yield"
            vehicle_considered = self.should_consider_vehicle(other_vehicle)
            # print_debug(f"[debug] vehicle_considered, basic_intersect_flag = {vehicle_considered, basic_intersect_flag}")
            if basic_intersect_flag:
                crossed = True
                reason = basic_reason
                action = basic_action

        # ordinary actors, which are not special
        CONSIDER_RADIUS = NORMAL_CONSIDER_RADIUS
        if other_vehicle.get('lane_relative_to_ego', 1) == 0:
            CONSIDER_RADIUS = SAME_LANE_CONSIDER_RADIUS
        if self.opposite_flag:
            CONSIDER_RADIUS = OPPOSITE_LANE_CONSIDER_RADIUS
        if other_vehicle['distance'] > CONSIDER_RADIUS:
            return crossed, command_str, reason, action

        # # Find the actor in front of the ego vehicle
        # if ego_vehicle['hazard_detected_40']:
        #     # If the car in front is in an intersection this might not work perfectly, however, given the labels
        #     # it can't be improved
        #     if ego_vehicle['hazard_detected_10']:
        #         actor_in_front = vehicles_by_id[ego_vehicle['affects_ego_10']]
        #     elif ego_vehicle['hazard_detected_15']:
        #         actor_in_front = vehicles_by_id[ego_vehicle['affects_ego_15']]
        #     elif ego_vehicle['hazard_detected_20']:
        #         actor_in_front = vehicles_by_id[ego_vehicle['affects_ego_20']]
        #     else:
        #         actor_in_front = vehicles_by_id[ego_vehicle['affects_ego_40']]
        
        # print(f"==== {self.current_measurement_path} ====")
        # if ego_vehicle['hazard_detected_10']:
        #     print(f"aff_10 = {vehicles_by_id[ego_vehicle['affects_ego_10']]['id']}")
        # if ego_vehicle['hazard_detected_15']:
        #     print(f"aff_15 = {vehicles_by_id[ego_vehicle['affects_ego_15']]['id']}")
        # if ego_vehicle['hazard_detected_20']:
        #     print(f"aff_20 = {vehicles_by_id[ego_vehicle['affects_ego_20']]['id']}")
        # if ego_vehicle['hazard_detected_40']:
        #     print(f"aff_40 = {vehicles_by_id[ego_vehicle['affects_ego_40']]['id']}")
        front_min_pos = INF_MAX
        actor_in_front = None
        for vehicle in vehicles_by_id.values():
            if vehicle['position'][0] > 0 and vehicle['lane_relative_to_ego'] == 0:
                if vehicle['position'][0] < front_min_pos:
                    front_min_pos = vehicle['position'][0]
                    actor_in_front = vehicle
        if actor_in_front is not None and actor_in_front['distance'] > ACTOR_IN_FRONT_CONSIDER_RADIUS:
            actor_in_front = None
        # if actor_in_front:
        #     print(f"actor_in_front = {actor_in_front['id']}, other_vehicle = {other_vehicle['id']}")

        # Check if the other vehicle is right in front of the ego vehicle
        if actor_in_front is not None and \
            actor_in_front['id'] == other_vehicle['id'] and \
            ego_vehicle['lane_type_str'] != 'Parking' and \
            (other_vehicle['distance'] < TOO_CLOSE_THRESHOLD or other_vehicle['speed'] < SLOW_VEHICLE_SPEED or \
            actor_in_front['speed'] < ego_vehicle['speed'] * SLOW_VEHICLE_RATIO or \
            actor_in_front.get('brake', 0.0) > VEHICLE_BRAKE_THRESHOLD or \
            actor_in_front.get('light_state', "None") == 'Brake'):
            # if vehicle is in front of ego vehicle on the same lane
            answer = f"Yes, the {other_vehicle_description} is right to the front of the ego vehicle, so the " +\
                                                        "ego vehicle should pay attention to not crash into it."
            crossed = True
            speed_str = "is driving"
            if actor_in_front['distance'] <= TOO_CLOSE_THRESHOLD:
                speed_str = "is driving too close"
            if abs(actor_in_front['speed']) <= max(SLOW_VEHICLE_SPEED, ego_vehicle['speed'] * SLOW_VEHICLE_RATIO):
                speed_str = "is driving slowly"
            if abs(actor_in_front['speed']) < STOP_VEHICLE_SPEED:
                speed_str = "has stopped"
            if actor_in_front.get('brake', 0.0) > VEHICLE_BRAKE_THRESHOLD:
                speed_str = "brakes"
            if actor_in_front.get('light_state', "None") == 'Brake':
                speed_str = "brakes"
            
            status_str = " too fast without decelerating"
            if abs(actor_in_front['speed']) < STOP_VEHICLE_SPEED and actor_in_front['distance'] <= STOP_FOR_STOPPED_VEHICLE_DISTANCE:
                status_str = ""
            reason = f"{speed_str} right to the front of the ego vehicle"
            if actor_in_front.get('light_state', "None") == 'Brake':
                reason += ", which can be seen from the brake lights"
            action = f"drives forward" + status_str
            if actor_in_front['lane_relative_to_ego'] == 0:
                action = f"drives forward along the current lane" + status_str
            
        
        # Check if the other vehicle is on the same road as the ego vehicle's next or next-next road
        elif (ego_distance_to_junction is not None and \
                                                ego_distance_to_junction < JUNCTION_CONSIDER_RADIUS) and \
            (other_vehicle['road_id'] in ego_vehicle['next_road_ids'] or \
            other_vehicle['road_id'] in ego_vehicle['next_next_road_ids_ego'] or same_future_road) and \
                other_vehicle['distance'] < JUNCTION_CONSIDER_RADIUS: # and not other_vehicle['same_direction_as_ego']:
            # handles vehicles that are in front but in an intersection
            
            # if scenario == 'BlockedIntersection' and \
            #                         other_vehicle['distance'] < 40 and theta < 45.0:
            #     # Check for the BlockedIntersection scenario
            #     loc_str = "will enter"
            #     if ego_vehicle['is_in_junction']:
            #         loc_str = "currently in"
            #     answer = f"Yes, the {other_vehicle_description} is behind the intersection on the road the ego " +\
            #                         f"vehicle {loc_str}, so the ego vehicle should pay attention to not crash into it."
            #     crossed = True
            #     reason = f"is behind the intersection on the road the ego vehicle {loc_str}"
            if other_vehicle['is_in_junction']:
                crossed = True
                if not basic_intersect_flag:
                    answer = f"Yes, the {other_vehicle_description} is inside the junction on the same " +\
                                "road as the ego vehicle, so the ego vehicle should pay attention to not crash into it."
                    reason = f"is inside the junction and going to enter the same road as the ego vehicle"
                    action = get_default_overlap_reason(other_vehicle)
                    if scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios:
                        reason = f"is near the highway intersection and is moving in the same direction as the ego vehicle"
                        action = get_default_overlap_reason(other_vehicle)

            elif theta < SAME_DIRECTION_MAX_THETA and not pointing_towards_junction: # and not 
                # (scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios):\
                if not basic_intersect_flag:
                    if (scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios):
                        crossed = False
                        if -HIGHWAY_INTERSECTION_SAME_DIRECTION_MAX_Y < other_vehicle['position'][1] < HIGHWAY_INTERSECTION_SAME_DIRECTION_MAX_Y:
                            crossed = True
                            reason = f"is driving in the same direction as the ego vehicle near highway junction"
                            action = get_default_overlap_reason(other_vehicle)
                    else:
                        crossed = True
                        answer = f"Yes, the {other_vehicle_description} is behind the intersection on the road the " +\
                                    "ego vehicle will enter, so the ego vehicle should pay attention to not crash into it."
                        reason = f"is behind the intersection on the road the ego vehicle will enter"
                        action = get_default_overlap_reason(other_vehicle)
                

        # Check if the other vehicle is in a junction and pointing towards the ego vehicle
        elif other_vehicle['is_in_junction'] and ego_distance_to_junction is not None and \
                                                ego_distance_to_junction < JUNCTION_CONSIDER_RADIUS:
            if not basic_intersect_flag:
                # intersection and vehicle points towards ego
                in_highway_flag = (scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios)
                # if (in_highway_flag and not pointing_towards_junction):
                #     # highway junction is too broad in carla
                #     # need restriction
                #     crossed = False
                # else:
                crossed = True
                desc_str = "is inside the upcoming junction"
                if in_highway_flag:
                    desc_str = "is near the upcoming highway junction"
                if ego_vehicle['is_in_junction']:
                    if in_highway_flag:
                        desc_str = "is near the highway junction"
                    desc_str = "is in the same intersection as the ego vehicle"
                answer = f"Yes, the {other_vehicle_description} is crossing the path of the ego vehicle."
                reason = f"{desc_str}"
                action = get_default_overlap_reason(other_vehicle)
                if other_vehicle_points_towards_ego:
                    reason = f"{desc_str} and pointing to the ego vehicle"
                    action = get_default_overlap_reason(other_vehicle)
                    if in_highway_flag:
                        reason = f"is near the highway intersection"
                        action = get_default_overlap_reason(other_vehicle)
                if (theta > 180 - OPPOSITE_DIRECTION_MAX_THETA and theta < 180 + OPPOSITE_DIRECTION_MAX_THETA) and command_int == 3:
                    crossed = False
                    answer = f"If the other vehicle keeps going straight, the routes will not cross."
                    reason = f"is in intersection but if this vehicle keeps going straight, the routes will not cross."
                    action = "does not keep going straight"
                    if in_highway_flag:
                        reason = f"is near the highway intersection"
                        action = get_default_overlap_reason(other_vehicle)

        # Check if the other vehicle is cutting into the ego vehicle's lane
        elif 'vehicle_cuts_in' in other_vehicle and other_vehicle['vehicle_cuts_in']:
            answer = f"Yes, the routes will cross since the {other_vehicle_description} is cutting into the " +\
                                                                                        "lane of the ego vehicle."
            crossed = True
            reason = "is cutting into the lane of the ego vehicle"
            action = "keeps driving forward without decelerating"
            if vehicle['speed'] >= ego_vehicle['speed'] * SLOW_VEHICLE_RATIO and vehicle['distance'] > BRAKE_FOR_SLOW_VEHICLE_DISTANCE:
                action = "drives forward too fast"
        
        # Check if the other vehicle is pointing towards the junction and the ego vehicle is close to the junction
        elif pointing_towards_junction and ego_distance_to_junction is not None and \
             ego_distance_to_junction < JUNCTION_CONSIDER_RADIUS and \
             theta > DIFFERENT_DIRECTION_MIN_THETA:
            if not basic_intersect_flag:
                answer = f"Yes, the {other_vehicle_description} might cross the path of the ego vehicle, depending " +\
                                                                            "on which way the vehicle is going to turn."
                crossed = True
                if other_vehicle['is_in_junction']:
                    reason = "is inside the junction"
                    action = get_default_overlap_reason(other_vehicle)
                    if scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios:
                        reason = f"is near the highway intersection"
                        action = get_default_overlap_reason(other_vehicle)
                else:
                    reason = "is going to enter the junction and may turn in any direction"
                    action = get_default_overlap_reason(other_vehicle)
                    if scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios:
                        reason = f"is near the highway intersection"
                        action = get_default_overlap_reason(other_vehicle)
            
        # Check if the ego vehicle is exiting a parking spot and the other vehicle is on the lane the ego vehicle 
        # wants to enter
        elif ego_vehicle['lane_type_str'] == 'Parking' and \
                                    other_vehicle['lane_relative_to_ego'] == -1 and other_vehicle['distance'] < 10:
            answer = f"Yes, because the ego vehicle wants to exit the parking spot and the " +\
                                    f"{other_vehicle_description} is on the lane the ego vehicle wants to enter."
            crossed = True
            reason = "is on the lane the ego vehicle wants to enter when the ego vehicle wants to exit the parking spot"
            action = "exit the parking spot now without waiting for it to pass"
        elif basic_intersect_flag == False:
            answer = f"No, the {other_vehicle_description} is not crossing paths with the ego vehicle."
            crossed = False
            reason = ""
            action = ""
        
        BACK_DANGER_DISTANCE = LANE_CHANGE_BACK_DANGER_DISTANCE
        # if scenario in self.highway_change_lane_scenarios or 'LaneChange' in scenario:
        if self.in_carla:
            danger_interval = LANE_CHANGE_DANGER_INTERVAL
            if scenario in ['YieldToEmergencyVehicle']:
                danger_interval = EMERGENCY_VEHICLE_DANGER_INTERVAL
            BACK_DANGER_DISTANCE = max(LANE_CHANGE_BACK_DANGER_DISTANCE, self.ideal_flow_speed * danger_interval)
        if scenario in ['ParkingExit']:
            BACK_DANGER_DISTANCE = LANE_CHANGE_PARKING_EXIT_BACK_DANGER_DISTANCE

        if other_vehicle['position'][0] > -3.0:
            cross_prefix = f"changes to "
            cross_subfix = " immediately"
        else:
            if other_vehicle['distance'] < BACK_DANGER_DISTANCE:
                cross_prefix = f"changes to "
                cross_subfix = " immediately"
            else:
                cross_prefix = f"changes to "
                cross_subfix = " but drives too slowly"

        # Check for lane changes of the ego vehicle
        if other_vehicle['lane_relative_to_ego'] == -1 and command_int == 5:
            answer = f"Yes, the {other_vehicle_description} is crossing paths with the ego vehicle because the " +\
                    f"ego vehicle does a lane change to the left onto the lane of the {other_vehicle_description}."
            crossed = True
            reason = "is on the left lane which the ego vehicle wants to change onto"
            action = cross_prefix + "the left lane" + cross_subfix
                
        elif other_vehicle['lane_relative_to_ego'] == 1 and command_int == 6:
            answer = f"Yes, the {other_vehicle_description} is crossing paths with the ego vehicle because the " +\
                    f"ego vehicle does a lane change to the right onto the lane of the {other_vehicle_description}."
            crossed = True
            reason = "is on the right lane which the ego vehicle wants to change onto"
            action = cross_prefix + "the right lane" + cross_subfix
        
        elif other_vehicle['lane_relative_to_ego'] == 0 and \
            other_vehicle['base_type'] == 'bicycle' and 0.0 < other_vehicle['position'][0] < 30.0 and is_vehicle_in_camera(self.CAMERA_FRONT, other_vehicle):
            answer = f"Yes, the {other_vehicle_description} is potentially crossing paths with the ego vehicle because the " +\
                    f"{other_vehicle_description} doesn't match the speed of ego vehicle."
            crossed = True
            reason = "doesn't match the speed of ego vehicle"
            action = "keeps driving forward without decelerating"
        
        elif other_vehicle['lane_relative_to_ego'] == 0 and self.opposite_flag is True and \
            0.0 < other_vehicle['position'][0]:
            answer = f"Yes, the {other_vehicle_description} is potentially crossing paths with the ego vehicle because the " +\
                    f"{other_vehicle_description} is driving towards the ego vehicle when the ego vehicle is driving in opposite lane."
            crossed = True
            reason = "is driving towards the ego vehicle when the ego vehicle is driving in opposite lane, this is extremely dangerous"
            action = "keeps driving in the opposite lane without changing back as soon as possible"
            
        # Check for scenarios involving acceleration lanes
        elif is_ego_on_highway and is_ego_in_accel_lane:
            # special case if ego is still on acceleration lane and the lane of the other vehicle is not considered
            # as the same road -> lane_relative_to_ego is None
            answer = f"The routes of the ego vehicle and the {other_vehicle_description} might cross as " +\
                f"the {other_vehicle_description} is on the highway and the ego vehicle is on the acceleration " +\
                                                                                f"lane about to enter the highway."
            crossed = True
            reason = "is on the highway and the ego vehicle is on the acceleration lane about to enter the highway"
            action = "keeps driving without yielding it when getting closer"
            if vehicle_is_too_dangerous(other_vehicle):
                action = "does not stop immediately"
        elif is_ego_on_highway and is_other_veh_in_accel_lane:
            answer = f"The routes of the ego vehicle and the {other_vehicle_description} might cross as " +\
                f"the {other_vehicle_description} is on the acceleration lane about to enter the highway, " +\
                f"potentially cutting into the lane of the ego vehicle."
            crossed = True
            reason = "is on the acceleration lane about to enter the highway, potentially cutting into the lane of the ego vehicle"
            action = "keeps driving without yielding it when getting closer"
            if vehicle_is_too_dangerous(other_vehicle):
                action = "does not stop immediately"

        # Check for the CrossingBicycleFlow scenario
        if other_vehicle['base_type'] == 'bicycle' \
                    and scenario == "CrossingBicycleFlow":
            # CrossingBicycleFlow scenario
            if command_int == 4:
                # Special case if the ego vehicle is not close enough to the junction and the command is still 
                # follow the road
                command_str = 'turns at the next intersection'
            answer = f"Yes, the bike lane on which the {other_vehicle_description} is currently riding on is " +\
                                            f"crossing paths with the ego vehicle if the ego vehicle {command_str}."
            crossed = True
            reason = f"is riding on a bike lane which crosses paths with the ego vehicle if the ego vehicle {command_str}"
            action = "keeps driving without yielding it when getting closer"
            if vehicle_is_too_dangerous(other_vehicle):
                action = "does not stop immediately"
        elif other_vehicle['base_type'] == 'bicycle' \
                    and scenario == "VehicleTurningRoute":
            answer = f"Yes, the {other_vehicle_description} will cross paths with the ego vehicle if " \
                        f"the ego vehicle {command_str}."
            crossed = True
            reason = f"is crossing ego vehicle's lane if the ego vehicle {command_str}"
            action = "keeps driving without yielding it when getting closer"
            if vehicle_is_too_dangerous(other_vehicle):
                action = "does not stop immediately"

        elif scenario == "HighwayCutIn" \
                    and other_vehicle['lane_relative_to_ego'] == 1:
            answer = f"Yes, the routes of the ego vehicle and the {other_vehicle_description} might cross as " +\
                f"the {other_vehicle_description} is on the acceleration lane, potentially cutting into the " +\
                f"lane of the ego vehicle."
            crossed = True
            reason = "is on the acceleration lane, potentially cutting into the lane of the ego vehicle"
            action = "keeps driving without yielding it when getting closer"
            if vehicle_is_too_dangerous(other_vehicle):
                action = "does not stop immediately"
        
        return crossed, command_str, reason, action

    def determine_vehicle_motion_status(other_vehicle_location_description, other_vehicle, ego_vehicle, opposite_flag,
                                        other_vehicle_description, qas_conversation_vehicle, object_tags, consider_vehicle):
        """
        Answers: "What is the moving status of {other_vehicle_location_description}?".
        
        Args:
            other_vehicle_location_description (str): Description of the other vehicle's location.
            other_vehicle (dict): Information about the other vehicle.
            other_vehicle_description (str): Description of the other vehicle.
            qas_conversation_vehicle (list): List of question-answer pairs for the vehicle.

        Returns:
            None
        """
        question = f"What is the moving status of {other_vehicle_location_description}?"

        # Determine motion status based on speed and vehicle type
        if other_vehicle['speed'] < STOP_VEHICLE_SPEED:
            motion_str = f"not moving"
        elif other_vehicle['speed'] < max(SLOW_VEHICLE_SPEED, ego_vehicle['speed'] * SLOW_VEHICLE_RATIO):
            if other_vehicle['base_type'] == 'bicycle':
                motion_str = f"moving slowly"
            else:
                motion_str = f"driving slowly"
        else:
            if other_vehicle['base_type'] == 'bicycle':
                motion_str = f"moving at a normal speed"
            else:
                motion_str = f"driving at a normal speed"
        
        direction_str = ""
        if other_vehicle['speed'] >= STOP_VEHICLE_SPEED:
            if other_vehicle['road_id'] == ego_vehicle['road_id']:
                if other_vehicle['lane_id'] * ego_vehicle['lane_id'] >= 0 and opposite_flag == False:
                    direction_str = "same"
                if other_vehicle['lane_id'] * ego_vehicle['lane_id'] < 0 and opposite_flag == True:
                    direction_str = "same"
                if other_vehicle['lane_id'] * ego_vehicle['lane_id'] >= 0 and opposite_flag == True:
                    direction_str = "opposite"
                if other_vehicle['lane_id'] * ego_vehicle['lane_id'] < 0 and opposite_flag == False:
                    direction_str = "opposite"

        # print(f"[debug] other_vehicle['speed'] = {other_vehicle['speed']}, other_vehicle['road_id'] = {other_vehicle['road_id']}, other_vehicle['lane_id'] = {other_vehicle['lane_id']}")
        # print(f"[debug] ego_vehicle['road_id'] = {ego_vehicle['road_id']}, ego_vehicle['lane_id'] = {ego_vehicle['lane_id']}")
        # print(f"[debug] direction_str = {direction_str}")

        if direction_str != "":
            motion_str = f"{motion_str} in the {direction_str} direction of the ego vehicle"
        answer = f"The {other_vehicle_description} is {motion_str}."
        

        # Add the question-answer pair to the conversation
        if consider_vehicle:
            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                    qid=16,
                                    chain=4,
                                    layer=2,
                                    qa_type='prediction',
                                    connection_up=[(4,3)],
                                    connection_down=[(4,0)],
                                    question=question,
                                    answer=answer,
                                    object_id=other_vehicle['id'],
                                    object_tags=object_tags)
        
        return motion_str

    def determine_vehicle_trajectory(other_vehicle_location_description, other_vehicle, other_vehicle_description,
                                                            qas_conversation_vehicle, object_tags, consider_vehicle):
        """
        Answer: "Where is {other_vehicle_location_description} going?".

        Args:
            other_vehicle_location_description (str): Description of the other vehicle's location.
            other_vehicle (dict): Information about the other vehicle.
            other_vehicle_description (str): Description of the other vehicle.
            qas_conversation_vehicle (list): List of question-answer pairs for the vehicle.

        Returns:
            None
        """

        question = f"Where is {other_vehicle_location_description} going?"
        answer = f"The {other_vehicle_description} is following the road."

        steer = other_vehicle['steer']

        # Determine trajectory based on steer angle
        if steer < -STEER_THRESHOLD:
            dir_phrase = f"turning left"
        elif steer < -SLIGHT_STEER_THRESHOLD:
            dir_phrase = f"turning slightly left"
        elif steer > STEER_THRESHOLD:
            dir_phrase = f"turning right"
        elif steer > SLIGHT_STEER_THRESHOLD:
            dir_phrase = f"turning slightly right"
        else:
            dir_phrase = f"going straight"

        # Check if the other vehicle is cutting into the ego vehicle's lane
        if 'vehicle_cuts_in' in other_vehicle:
            if other_vehicle['vehicle_cuts_in']:
                dir_phrase = f"cutting into the lane of the ego vehicle"

        answer = f"The {other_vehicle_description} is {dir_phrase}."

        # Add the question-answer pair to the conversation
        if consider_vehicle:
            self.add_qas_questions(qa_list=qas_conversation_vehicle,
                                    qid=17,
                                    chain=4,
                                    layer=1,
                                    qa_type='prediction',
                                    connection_up=[(4,3)],
                                    connection_down=[(4,0)],
                                    question=question,
                                    answer=answer,
                                    object_id=other_vehicle['id'],
                                    object_tags=object_tags)
        
        return dir_phrase

    def determine_other_vehicle_position(other_vehicle_location_description, other_vehicle, ego_vehicle,
                                    is_ego_on_highway, is_ego_in_accel_lane, is_ego_in_exit_lane,
                                    other_vehicle_description, is_ego_in_entry_lane, ego_about_to_exit_highway,
                                    scenario):
        """
        Answers: "Where on the road is {other_vehicle_location_description} located?".

        Args:
            other_vehicle_location_description (str): Description of the other vehicle's location.
            other_vehicle (dict): Information about the other vehicle.
            ego_vehicle (dict): Information about the ego vehicle.
            is_ego_on_highway (bool): True if the ego vehicle is on a highway.
            is_ego_in_accel_lane (bool): True if the ego vehicle is in an acceleration lane.
            is_ego_in_exit_lane (bool): True if the ego vehicle is in an exit lane.
            other_vehicle_description (str): Description of the other vehicle.
            is_ego_in_entry_lane (bool): True if the ego vehicle is in an entry lane.
            ego_about_to_exit_highway (bool): True if the ego vehicle is about to exit the highway.
            scenario (str): The current scenario.
            qas_conversation_vehicle (list): List of question-answer pairs for the vehicle.

        Returns:
            pointing_towards_junction (bool or None): True if the other vehicle is pointing towards the junction, 
                False if pointing away from the junction, or None if the direction is unknown.
            position_str (str): describe the position of the vehicle
        """

        scenario = scenario.split("_")[0]
        question = f"Where on the road is {other_vehicle_location_description} located?"
        answer = ''
        pos = other_vehicle['position']

        num_lane_map = {
            0: 'same lane as',
            -1: 'the first lane left from',
            -2: 'the second lane left from',
            -3: 'the third lane left from',
            -4: 'the fourth lane left from',
            1: 'the first lane right from',
            2: 'the second lane right from',
            3: 'the third lane right from',
            4: 'the fourth lane right from'
        }

        # Check if the other vehicle is on the same road as the ego vehicle
        same_road = other_vehicle['same_road_as_ego']

        # Check if the other vehicle is moving in the same direction as the ego vehicle
        same_direction = other_vehicle['same_direction_as_ego']
        pointing_towards_junction = None

        # Determine the other vehicle's orientation relative to the ego vehicle
        orientation_relative_to_ego = other_vehicle['yaw']
        orientation_relative_to_ego = orientation_relative_to_ego * 180 / np.pi
        if orientation_relative_to_ego > 180.0:
            orientation_relative_to_ego -= 360.0
        
        # Categorize the orientation into 4 bins: leftwards, straight, rightwards, oncoming
        if -135 < orientation_relative_to_ego < -45:
            orientation_str = 'is pointing leftwards'
        elif 45 < orientation_relative_to_ego < 135:
            orientation_str = 'is pointing rightwards'
        elif 135 < orientation_relative_to_ego or orientation_relative_to_ego < -135:
            orientation_str = 'is pointing towards the ego vehicle'
        else:
            orientation_str = 'is pointing in the same direction as the ego vehicle'

        position_str = orientation_str
        in_junction_str = 'inside the junction'

        if scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios:
            in_junction_str = 'near the highway intersection'
        if scenario in self.enter_highway_scenarios and scenario != "MergerIntoSlowTrafficV2":
            in_junction_str = 'near the merging area'
        if scenario in self.leave_highway_scenarios:
            in_junction_str = 'near the highway exit erea'

        # Handle cases where the other vehicle is in a junction or on another entry of the junction
        if other_vehicle['is_in_junction'] and (other_vehicle['junction_id'] == ego_vehicle['next_junction_id'] or \
                    other_vehicle['junction_id'] == ego_vehicle['junction_id'] or \
                    (ego_vehicle['junction_id'] == -1 and ego_vehicle['next_junction_id'] == -1)):
            if is_ego_on_highway and (is_ego_in_accel_lane or is_ego_in_exit_lane):
                # Handle cases where the ego vehicle is in the merging or exit area of the highway
                if is_ego_in_accel_lane: 
                    lane_str='merging area'
                if is_ego_in_exit_lane: 
                    lane_str='exit area'
                if other_vehicle['same_road_as_ego'] and other_vehicle['same_direction_as_ego'] and \
                                                                        other_vehicle['lane_relative_to_ego'] == 0:
                    answer = f"The {other_vehicle_description} is in the {lane_str} of the highway in front " +\
                                                                                        f"of the ego vehicle."
                    position_str = f"is in the {lane_str} of the highway in front " +\
                                                                                        f"of the ego vehicle"
                elif other_vehicle['lane_relative_to_ego'] is not None and abs(other_vehicle['lane_relative_to_ego']) <= 4:
                    answer = f"The {other_vehicle_description} is close to the {lane_str} but on the leftmost " +\
                                                                                            f"lane of the highway."
                    position_str = f"is close to the {lane_str} but on the leftmost " +\
                                                                                            f"lane of the highway"
                else:
                    answer = f"The {other_vehicle_description} is on the highway near the {lane_str}."
                    position_str = f"is on the highway near the {lane_str}"
            elif is_ego_in_entry_lane:
                answer = f"The {other_vehicle_description} is on the lane that leads to the highway."
                position_str = f"is on the lane that leads to the highway"
            elif ego_about_to_exit_highway:
                answer = f"The {other_vehicle_description} is on the exit lane of the highway."
                position_str = f"is on the exit lane of the highway"
            elif is_ego_on_highway:
                answer = f"The {other_vehicle_description} is on the highway."
                position_str = f"is on the highway"
            elif other_vehicle['road_id'] not in ego_vehicle['next_road_ids']:
                answer = f"The {other_vehicle_description} is {in_junction_str} and {orientation_str}."
                position_str = f"is {in_junction_str} and {orientation_str}"
            else:
                answer = f"The {other_vehicle_description} is {in_junction_str} and {orientation_str}."
                position_str = f"is {in_junction_str} and {orientation_str}"

            # Handle the MergerIntoSlowTrafficV2 scenario
            if scenario == "MergerIntoSlowTrafficV2": # or scenario == "MergerIntoSlowTraffic":
                if ego_vehicle['num_lanes_same_direction'] == 1 and other_vehicle['same_road_as_ego'] or \
                                (ego_vehicle['num_lanes_same_direction'] - ego_vehicle['ego_lane_number']-1 == 0 and \
                                                ego_vehicle['distance_to_junction'] is not None and ego_vehicle['distance_to_junction']<25 and \
                                                (other_vehicle['road_id'] in ego_vehicle['next_road_ids'] or \
                                                other_vehicle['road_id'] == ego_vehicle['road_id'] or \
                                                other_vehicle['road_id'] in ego_vehicle['next_next_road_ids_ego'])):
                    answer = f"The {other_vehicle_description} is on the exit lane of the highway."
                    position_str = f"is on the exit lane of the highway"
                elif ego_vehicle['num_lanes_same_direction'] == 1 and not other_vehicle['same_road_as_ego'] or \
                                    (ego_vehicle['num_lanes_same_direction'] > 1 and \
                                    (ego_vehicle['is_in_junction'] or ego_vehicle['distance_to_junction'] < 25)):
                    answer = f"The {other_vehicle_description} is on the highway near the exit area."
                    position_str = f"is on the highway near the exit area"
                else:
                    answer = f"The {other_vehicle_description} is on the highway close to the merging area."
                    position_str = f"is on the highway close to the merging area"

        # Handle cases where the other vehicle is not in a junction and not on the same road as the ego vehicle
        elif not other_vehicle['is_in_junction'] and (other_vehicle['road_id'] != ego_vehicle['road_id'] and \
                                                    other_vehicle['road_id'] not in ego_vehicle['next_road_ids']):
            # Determine if the other vehicle is pointing towards or away from the junction
            if ego_vehicle['junction_id'] == -1 or other_vehicle['junction_id'] == -1:
                # print(f'[debug] path = {self.current_measurement_path}, id = {other_vehicle["id"]}, orientation_rel = {orientation_relative_to_ego}, position = {pos}')
                if pos[1] < -8 and orientation_relative_to_ego > 45 and orientation_relative_to_ego < 135:
                    to_or_away_junction = "is pointing towards the junction"
                    pointing_towards_junction = True
                elif pos[1] > 8 and orientation_relative_to_ego < -45 and orientation_relative_to_ego > -135:
                    to_or_away_junction = "is pointing towards the junction"
                    pointing_towards_junction = True
                elif pos[1] < -8 and orientation_relative_to_ego < -45 and orientation_relative_to_ego > -135:
                    to_or_away_junction = "is pointing away from the junction"
                    pointing_towards_junction = False
                elif pos[1] > 8 and orientation_relative_to_ego > 45 and orientation_relative_to_ego < 135:
                    to_or_away_junction = "is pointing away from the junction"
                    pointing_towards_junction = False
                elif pos[1] < 8 and pos[1] > -8 and orientation_relative_to_ego > 135 or \
                                                                                orientation_relative_to_ego < -135:
                    to_or_away_junction = "is pointing towards the junction"
                    pointing_towards_junction = True
                elif pos[1] < 8 and pos[1] > -8 and orientation_relative_to_ego < 45 and \
                                                                                orientation_relative_to_ego > -45:
                    to_or_away_junction = "is pointing away from the junction"
                    pointing_towards_junction = False
                else:
                    if -135 < orientation_relative_to_ego < -45:
                        orientation_str = 'is pointing leftwards'
                    elif 45 < orientation_relative_to_ego < 135:
                        orientation_str = 'is pointing rightwards'
                    elif 135 < orientation_relative_to_ego or orientation_relative_to_ego < -135:
                        orientation_str = 'is pointing towards the ego vehicle'
                    else:
                        orientation_str = 'is pointing in the same direction as the ego vehicle'
                    # to_or_away_junction = "is pointing in an unknown direction"
                    to_or_away_junction = orientation_str
                    pointing_towards_junction = False

            elif other_vehicle['next_junction_id'] == ego_vehicle['next_junction_id'] or \
                                                    other_vehicle['next_junction_id'] == ego_vehicle['junction_id']:
                to_or_away_junction = "is pointing towards the junction"
            # Away from junction
            elif other_vehicle['next_junction_id'] != ego_vehicle['next_junction_id'] and \
                                                    other_vehicle['next_junction_id'] != ego_vehicle['junction_id']:
                to_or_away_junction = "is pointing away from the junction"

            # Determine the direction of the other vehicle relative to the junction
            if pos[1] < -JUNCTION_POS_OFFSET:
                direction_junction = "on the left side of the junction"
            elif pos[1] > JUNCTION_POS_OFFSET:
                direction_junction = "on the right side of the junction"
            elif not other_vehicle['same_road_as_ego']:
                direction_junction = "on the opposite side of the junction"
            else:
                direction_junction = "near the junction"
                # raise ValueError(f"Unknown position of vehicle {other_vehicle['id']}.")

            # Add information if the other vehicle is on a bike lane
            bike_lane_str = ''
            if other_vehicle['lane_type_str'] == 'Biking':
                bike_lane_str = ' on the bike lane'

            answer = f"The {other_vehicle_description} is {direction_junction}{bike_lane_str} and " +\
                                                                                        f"{to_or_away_junction}."
            position_str = f"is {direction_junction}{bike_lane_str} and {to_or_away_junction}"

            # Handle cases where the ego vehicle is on the highway
            if is_ego_on_highway and ego_vehicle['road_id'] == other_vehicle['road_id']:
                if other_vehicle['lane_relative_to_ego'] is not None:
                    if abs(other_vehicle['lane_relative_to_ego']) <= 4:
                        answer = f"The {other_vehicle_description} is driving on {num_lane_map[other_vehicle['lane_relative_to_ego']]} " +\
                            "the ego vehicle on the highway."
                        position_str = f"is driving on {num_lane_map[other_vehicle['lane_relative_to_ego']]} " +\
                            "the ego vehicle on the highway."
                    else:   
                        answer = f"The {other_vehicle_description} is driving on the highway."
                        position_str = f"is driving on the highway"
                else:   
                    answer = f"The {other_vehicle_description} is driving on the highway."
                    position_str = f"is driving on the highway"

        # Handle cases where the ego vehicle is in a junction, and the other vehicle is on the road
        # the ego vehicle will enter
        elif (ego_vehicle['is_in_junction'] and other_vehicle['road_id'] in ego_vehicle['next_road_ids']):
            if is_ego_on_highway and scenario in self.enter_highway_scenarios:
                answer = f"The {other_vehicle_description} is on the highway."
                position_str = f"is on the highway"
            else:
                answer = f"The {other_vehicle_description} is after the junction on the road the ego vehicle " +\
                                                                            f"will enter. It {orientation_str}."
                position_str = f"is after the junction on the road the ego vehicle " +\
                                                                            f"will enter. It {orientation_str}"

        # Handle cases where neither vehicle is in a junction, and both are on the same road
        elif not other_vehicle['is_in_junction'] and same_road:
            value = int(other_vehicle['lane_relative_to_ego'])
            if self.opposite_flag:
                value = -value
            value_num = value
            s_or_no_s = 's' if abs(value) > 1 else ''
            if value == 0:
                # value = 'on the same lane as'
                pass
            elif value > 0:
                value = f"{number_to_word(abs(value))} lane{s_or_no_s} to the right of"
            else: # value < 0
                value = f"{number_to_word(abs(value))} lane{s_or_no_s} to the left of"

            bike_lane_str = ''
            if other_vehicle['lane_type_str'] == 'Biking':
                bike_lane_str = ' on the bike lane'
            special_lane_flag = False
            special_lane_str = ''
            if other_vehicle['lane_type_str'] == 'Parking':
                special_lane_flag = True
                special_lane_str = ' in the parking area'
            if other_vehicle['lane_type_str'] == 'Shoulder':
                special_lane_flag = True
                special_lane_str = ' on the shoulder'

            moving_action = 'stops' if other_vehicle['speed'] < STOP_VEHICLE_SPEED else 'driving'
            same_side_str = 'in the same direction'
            opp_side_str = 'in the opposite direction'
            if self.opposite_flag:
                same_side_str = 'on the same side, but the vehicle is driving on wrong side'
                opp_side_str = 'on the other side, but the vehicle is driving on wrong side'

            if same_direction:
                if value == 0:
                    answer = f"The {other_vehicle_description} is on the same road {moving_action} on the " +\
                                                                    f"lane of the ego vehicle."
                    position_str = f"is on the same road {moving_action} on the " +\
                                                                    f"lane of the ego vehicle"
                else:
                    answer = f"The {other_vehicle_description} is on the same road {moving_action} {same_side_str}." +\
                                                f" It is{bike_lane_str} {value} the ego vehicle."
                    position_str = f"is on the same road {moving_action} {same_side_str}." +\
                                                f" It is{bike_lane_str} {value} the ego vehicle"
                    if special_lane_flag:
                        position_str = f"is on the same road {moving_action} {same_side_str}." +\
                                                f" It is{special_lane_str} {value} the ego vehicle"
                if special_lane_flag and 0 <= value_num <= 0: # 1:
                        position_str = f"is on the same road {moving_action}{special_lane_str}"
            else:
                answer = f"The {other_vehicle_description} is on the same road {moving_action} {opp_side_str}." +\
                                                        f" It is{bike_lane_str} {value} the ego vehicle."
                position_str = f"is on the same road {moving_action} {opp_side_str}." +\
                                                        f" It is{bike_lane_str} {value} the ego vehicle"
                if special_lane_flag:
                        position_str = f"is on the same road {moving_action} {opp_side_str}." +\
                                                f" It is{special_lane_str} {value} the ego vehicle"

            if is_ego_in_entry_lane:
                answer = f"The {other_vehicle_description} is in the same lane leading to the highway as " +\
                                                                                            f"the ego vehicle."
                position_str = f"is in the same lane leading to the highway as the ego vehicle"

        
        # Handle the HighwayCutIn scenario
        if scenario == "HighwayCutIn" and \
                                                    other_vehicle['road_id'] != ego_vehicle['road_id']:
            # Currently we can't differentiate between acceleration lane and entry lane
            answer = f"The {other_vehicle_description} is on the acceleration lane of the highway to the right " +\
                                                                                            f"of the ego vehicle."
            position_str = f"is on the acceleration lane of the highway merging to the ego vehicle's lane"

        # Add the question-answer pair to the conversation
        # if consider_vehicle:
        #     self.add_qas_questions(qa_list=qas_conversation_vehicle,
        #                             chain=4,
        #                             layer=0,
        #                             qa_type='perception',
        #                             connection_up=[(4,1), (4,2), (4,3)],
        #                             connection_down=[(3,0),(3,2),(3,3)],
        #                             question=question,
        #                             answer=answer,
        #                             object_id=other_vehicle['id'],
        #                             object_tags=object_tags)
    
        return pointing_towards_junction, position_str

    scenario = scenario.split('_')[0]
    # main contents of this function starts here 
    qas_conversation_vehicle = []

    # Initialize the distance to the next junction for the ego vehicle
    ego_distance_to_junction = ego_vehicle['distance_to_junction']
    if ego_distance_to_junction is None:
        ego_distance_to_junction = INF_MAX # Set a default value if distance to junction is not available

    speed_limit_kmh = self.current_speed_limit
    # this value only used to indicate specified scenario
        
    # Flags to indicate if the ego vehicle is in an acceleration lane, exit lane, or entry lane
    is_ego_on_highway = False
    is_ego_in_accel_lane = False
    is_other_veh_in_accel_lane = False
    is_ego_in_exit_lane = False
    is_ego_in_entry_lane = False

    # Flag to indicate if the ego vehicle is about to exit the highway
    ego_about_to_exit_highway = False

    # List of scenario names that are considered highway scenarios
    highway_scenario_names = [
        # "EnterActorFlow", 
        # "EnterActorFlowV2", 
        "HighwayCutIn", 
        "HighwayExit", 
        "MergerIntoSlowTraffic",
        "MergerIntoSlowTrafficV2",
        "InterurbanActorFlow",
        "InterurbanAdvancedActorFlow"
        # "YieldToEmergencyVehicle",
    ]

    # Checks depend on scenario type and set flags accordingly
    if ego_vehicle['is_in_junction'] or ego_vehicle['num_lanes_same_direction'] > 1:
        if scenario == "HighwayCutIn" or scenario == "InterurbanAdvancedActorFlow":
            is_ego_on_highway = True
            # if ego_vehicle['is_in_junction'] or ego_distance_to_junction < 25:
            #     is_other_veh_in_accel_lane = True
        elif scenario == "HighwayExit" or scenario == "MergerIntoSlowTrafficV2" or \
            scenario == "MergerIntoSlowTraffic" or scenario == 'InterurbanActorFlow': 
            is_ego_on_highway = True
            if ego_vehicle['is_in_junction'] or ego_distance_to_junction < 25:
                is_ego_in_exit_lane = True
            if ego_vehicle['num_lanes_same_direction'] - ego_vehicle['ego_lane_number'] - 1 == 0 and \
                        current_measurement['command'] == 6 and ego_distance_to_junction < 40:
                ego_about_to_exit_highway = True
        elif scenario in highway_scenario_names: # and speed_limit_kmh > 50:
            is_ego_on_highway = True
            if scenario == 'MergerIntoSlowTraffic' and ego_vehicle['num_lanes_same_direction'] == 1 and \
                                                                    ego_vehicle['num_lanes_opposite_direction'] == 1:
                is_ego_in_entry_lane = True
                is_ego_in_accel_lane = False
            elif scenario == 'MergerIntoSlowTraffic' and ego_vehicle['num_lanes_same_direction'] > 1:
                is_ego_in_entry_lane = False
                is_ego_in_accel_lane = False
            elif ego_vehicle['is_in_junction'] or ego_distance_to_junction < 25:
                is_ego_in_accel_lane = True
            elif ego_vehicle['num_lanes_same_direction'] == 1 and ego_vehicle['num_lanes_opposite_direction'] == 0:
                is_ego_in_entry_lane = True
        # print(f"scenaro = {scenario}, is_ego_on_highway = {is_ego_on_highway}")

    self.ego_command_str = "follow the road"

    for vehicle in other_vehicles:
        is_dangerous = False
        # Check if the vehicle should be considered based on some criteria
        consider_vehicle = self.should_consider_vehicle(vehicle)

        # Special case: opposite vehicle, extremely dangerous!
        if self.opposite_flag == True:
            if vehicle['lane_relative_to_ego'] == 0 and (0 <= vehicle['position'][0] <= OPPOSITE_LANE_CONSIDER_RADIUS):
                consider_vehicle = True
                is_dangerous = True
        
        command_int = get_command_int_by_current_measurement(current_measurement, ego_vehicle)
        if ego_vehicle['is_in_junction'] and command_int not in [1, 2, 3]:
            if ego_vehicle['steer'] < -TURN_STEER_THRESHOLD: command_int = 1
            elif ego_vehicle['steer'] > TURN_STEER_THRESHOLD: command_int = 2
        
        if scenario in self.merging_scenarios and command_int in [1, 2, 3] and \
           ego_vehicle['distance_to_junction'] < JUNCTION_EXTEND_DISTANCE:
            basic_cross, _, _, _, _ = determine_basic_vector_crossing(command_int=command_int,
                                                                      other_vehicle=vehicle,
                                                                      whitelist=self.accelerate_white_list,
                                                                      following_list=self.vehicle_ids_following_ego,
                                                                      first_appear_history=self.first_appear_intervals)
            if vehicle['id'] in self.accelerate_black_list and \
            vehicle['approaching_dot_product'] < 0 and \
            vehicle['distance'] < self.accelerate_black_list[vehicle['id']]:
                basic_cross = True
            if basic_cross:
                consider_vehicle = True
                is_dangerous = True

        # # Get the position of the ego vehicle (every other vehicles positions are in the local coordinate system
        # # of the ego vehicle)
        # pos_ego = np.array([0,0,0])
        
        # # Get the position of the current vehicle
        # pos_vehicle = np.array(vehicle['position'])

        # # Calculate the angle between the vehicle and the ego vehicle
        # angle_rad = np.arctan2(pos_vehicle[1] - pos_ego[1], pos_vehicle[0] - pos_ego[0])
        # angle_deg = angle_rad * 180. / np.pi
        # angle_deg = angle_deg % 360. # Normalize the angle to [0, 360]

        # # Get the yaw angle (heading) of the vehicle
        # vehicle_heading_angle_rad = vehicle['yaw']
        # vehicle_heading_angle_deg = vehicle_heading_angle_rad * 180 / np.pi
        # vehicle_heading_angle_deg = vehicle_heading_angle_deg % 360. # Normalize the angle to [0, 360]

        other_vehicle_points_towards_ego, vehicle_heading_angle_deg = is_vehicle_pointing_towards_ego(vehicle['position'], vehicle['yaw'], 45)

        important_object_str, vehicle_description, vehicle_location_description = get_vehicle_str(vehicle)

        if consider_vehicle:
            important_objects.append(important_object_str)

        # Project the vehicle's bounding box points onto the image plane
        # projected_points, projected_points_meters = project_all_corners(vehicle, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
        project_dict = get_project_camera_and_corners(vehicle, self.CAM_DICT)
        # projected_points_meters[:, 2] -= vehicle['position'][2]

        # Generate a unique key and value for the vehicle object
        key, value = self.generate_object_key_value(
            id=vehicle['id'],
            category='Vehicle',
            visual_description=vehicle_description,
            detailed_description=vehicle_location_description,
            object_count=len(key_object_infos),
            is_dangerous=is_dangerous,
            obj_dict=vehicle,
            projected_dict=project_dict
        )

        if consider_vehicle:
            key_object_infos[key] = value

        object_tags = [key]
        vehicle_description += f"({object_tags})"
        vehicle_location_description += f"({object_tags})"

        res = determine_other_vehicle_position(vehicle_location_description, 
                                                vehicle, ego_vehicle, 
                                                is_ego_on_highway, 
                                                is_ego_in_accel_lane, 
                                                is_ego_in_exit_lane, 
                                                vehicle_description, 
                                                is_ego_in_entry_lane,
                                                ego_about_to_exit_highway, 
                                                scenario)
        
        pointing_towards_junction, position_str = res
        
        dir_phrase = determine_vehicle_trajectory(vehicle_location_description, vehicle, 
                                            vehicle_description, 
                                            qas_conversation_vehicle,
                                            object_tags, consider_vehicle)
        
        motion_str = determine_vehicle_motion_status(vehicle_location_description, 
                                                vehicle, 
                                                ego_vehicle,
                                                self.opposite_flag,
                                                vehicle_description, 
                                                qas_conversation_vehicle,
                                                object_tags, consider_vehicle)
        
        # add it later
        # if consider_vehicle:
        #     question = f"What is the rough moving speed and moving direction of {vehicle_location_description}?"
        #     answer = f"The {vehicle_description} is {motion_str}, {dir_phrase}."

        #     self.add_qas_questions(qa_list=qas_conversation_vehicle,
        #                             chain=4,
        #                             layer=1,
        #                             qa_type='prediction',
        #                             connection_up=[(4,3)],
        #                             connection_down=[(4,0)],
        #                             question=question,
        #                             answer=answer,
        #                             object_id=vehicle['id'],
        #                             object_tags=object_tags)
            
        #     question = f"What is the exact moving speed and moving direction of {vehicle_location_description}?"
        #     answer = f"The {vehicle_description} is driving at the speed of {vehicle['speed']:.1f} m/s, {dir_phrase}."

        #     self.add_qas_questions(qa_list=qas_conversation_vehicle,
        #                             chain=4,
        #                             layer=1,
        #                             qa_type='prediction',
        #                             connection_up=[(4,3)],
        #                             connection_down=[(4,0)],
        #                             question=question,
        #                             answer=answer,
        #                             object_id=vehicle['id'],
        #                             object_tags=object_tags)

        res = determine_path_crossing(current_measurement, 
                                      ego_vehicle, 
                                      vehicle_location_description, 
                                      vehicle, 
                                      vehicles_by_id, 
                                      vehicle_description, 
                                      scenario, 
                                      ego_distance_to_junction, 
                                      other_vehicle_points_towards_ego, 
                                      vehicle_heading_angle_deg,
                                      pointing_towards_junction, 
                                      is_ego_on_highway, 
                                      is_ego_in_accel_lane, 
                                      is_other_veh_in_accel_lane)

        crossed, self.ego_command_str, cross_reason, cross_action = res

        if vehicle_location_description.startswith("the "):
            vehicle_location_description = vehicle_location_description[4:]
        
        cur_dict = {
            "obj_id": vehicle['id'],
            "obj_tags": object_tags,
            "consider": consider_vehicle,
            "position_str": position_str,
            "dir": dir_phrase,
            "motion": motion_str,
            "speed": vehicle['speed'],
            "distance": vehicle['distance'],
            "description": vehicle_location_description,
            "cross_flag": crossed,
            "cross_reason": cross_reason,
            "cross_action": cross_action
        }

        self.all_vehicles_info[vehicle['id']] = cur_dict

    return qas_conversation_vehicle, important_objects, key_object_infos