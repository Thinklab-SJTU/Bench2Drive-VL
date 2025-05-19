from .offline_map_calculations import *
from .graph_utils import *
from .hyper_params import *
from io_utils import print_debug
import json

class SpeedCommand:
    def __init__(self):
        self.keep = 'KEEP'
        self.accelerate = 'ACCELERATE'
        self.decelerate = 'DECELERATE'
        self.stop = 'STOP'

class DirectionCommand:
    def __init__(self):
        self.follow = 'FOLLOW_LANE'
        self.left_change = 'CHANGE_LANE_LEFT'
        self.right_change = 'CHANGE_LANE_RIGHT'
        self.straight = 'GO_STRAIGHT'
        self.left_turn = 'TURN_LEFT'
        self.right_turn = 'TURN_RIGHT'
        self.left_deviate = 'DEVIATE_LEFT'
        self.right_deviate = 'DEVIATE_RIGHT'

def answer_behaviour_questions(self, ego_vehicle, other_vehicles, scene_data, current_measurement, scenario,
                                            important_objects, key_object_infos,
                                            final_lane_change_flag, final_change_dir, changed_for_real,
                                            final_brake_flag, final_stop_flag):
    
    qas_conversation_behaviour = []
    spd_cmd = SpeedCommand()
    dir_cmd = DirectionCommand()

    # high-level command
    scenario = scenario.split('_')[0]

    # Map command integers to their corresponding string descriptions
    command_int = get_command_int_by_current_measurement(current_measurement=current_measurement,
                                                         ego_vehicle=ego_vehicle)

    lane_curvature = get_lane_curvature(self.map, ego_vehicle['location'], CURVATURE_CHECK_DISTANCE, self.opposite_flag)
    curve_str = ""
    if lane_curvature == "left":
        curve_str = " Additionaly, since the current lane curves to the left, " +\
            "the ego vehicle should steer left based on the curvature shown in the image."
    if lane_curvature == "right":
        curve_str = " Additionaly, since the current lane curves to the right, " +\
            "the ego vehicle should steer right based on the curvature shown in the image."

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
    action_map = {
        1: 'continue driving along the current lane, then turn left at the intersection',
        2: 'continue driving along the current lane, then turn right at the intersection',
        3: 'continue driving along the current lane, then drive straight at the intersection',
        4: 'continue driving along the current lane',
        # 5: f'do a lane change to the left in around {int(command_distance)} m',
        # 6: f'do a lane change to the right in around {int(command_distance)} m',
        5: f'do a lane change to the left',
        6: f'do a lane change to the right',
    }
    dir_cmd_map = {
        1: dir_cmd.follow,
        2: dir_cmd.follow,
        3: dir_cmd.follow,
        4: dir_cmd.follow,
        5: dir_cmd.follow, # listen to lane_change
        6: dir_cmd.follow, # listen to lane_change
    }
    command_str = command_map[command_int]
    final_dir_cmd = dir_cmd_map[command_int]
    final_spd_cmd = spd_cmd.keep

    print_debug(f"[debug] command_int = {command_int}")

    drive_along_prefix = "continue driving along the current lane, "

    abstract_question = "What is the correct action for the ego vehicle to take now?"
    action_prefix = "The ego vehicle should "
    abstract_answer = action_prefix + action_map[command_int]

    angle_exit_exists = ego_vehicle['angle_exit'] is not None
    exit_ego_cord = transform_to_ego_coordinates([current_measurement['junction_exit_wp_x'],
                                                  current_measurement['junction_exit_wp_y'],
                                                  ego_vehicle['location'][2]],
                                                  ego_vehicle['world2ego'])
    exit_distance = math.sqrt(exit_ego_cord[0]**2 + exit_ego_cord[1]**2)
    if exit_distance < 0.5: # too close, this angle is not not accurate
        angle_exit_exists = False
    # if exit_ego_cord[0] < 0.0: # fell behind
    #     angle_exit_exists = False

    if ego_vehicle['is_in_junction'] or ego_vehicle.get('distance_to_junction', INF_MAX) < JUNCTION_EXTEND_DISTANCE:
        if command_int == 1:
            abstract_answer = action_prefix + 'continue turning left at the current intersection'
            final_dir_cmd = dir_cmd.left_turn
            if angle_exit_exists and ego_vehicle['angle_exit'] > TURNING_STOP_ANGLE:
                abstract_answer = action_prefix + "steer to the right because it has already steered too much to the left " +\
                                                  "if it wants to turn left at the current intersection"
                final_dir_cmd = dir_cmd.right_turn
            if angle_exit_exists and -TURNING_STOP_ANGLE < ego_vehicle['angle_exit'] < TURNING_STOP_ANGLE:
                abstract_answer = action_prefix + 'drive forward to get out of the intersection'
                final_dir_cmd = dir_cmd.straight
        if command_int == 2:
            abstract_answer = action_prefix + 'continue turning right at the current intersection'
            final_dir_cmd = dir_cmd.right_turn
            if angle_exit_exists and ego_vehicle['angle_exit'] < -TURNING_STOP_ANGLE:
                abstract_answer = action_prefix + "steer to the left because it has already steered too much to the right " +\
                                                  "if it wants to turn right at the current intersection"
                final_dir_cmd = dir_cmd.left_turn
            if angle_exit_exists and -TURNING_STOP_ANGLE < ego_vehicle['angle_exit'] < TURNING_STOP_ANGLE:
                abstract_answer = action_prefix + 'drive forward to get out of the intersection'
                final_dir_cmd = dir_cmd.straight
        if command_int == 3:
            abstract_answer = action_prefix + 'continue driving straight at the current intersection'
            final_dir_cmd = dir_cmd.straight
            if angle_exit_exists and ego_vehicle['angle_exit'] > TURNING_STOP_ANGLE:
                abstract_answer = action_prefix + "steer to the right because it steered too much to the left " +\
                                                  "if it wants to go straight at the current intersection"
                final_dir_cmd = dir_cmd.right_turn
            if angle_exit_exists and ego_vehicle['angle_exit'] < -TURNING_STOP_ANGLE:
                abstract_answer = action_prefix + "steer to the left because it steered too much to the right " +\
                                                  "if it wants to go straight at the current intersection"
                final_dir_cmd = dir_cmd.left_turn

    if scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios:
        if scenario in ['InterurbanAdvancedActorFlow']:
            if command_int == 1:
                abstract_answer = action_prefix + drive_along_prefix + 'then turn left to join the main traffic'
            if command_int == 2:
                abstract_answer = action_prefix + drive_along_prefix + 'then turn right to join the main traffic'
            if command_int == 3:
                abstract_answer = action_prefix + drive_along_prefix + 'then go straight to join the main traffic'
            if ego_vehicle['is_in_junction']:
                if command_int == 1:
                    abstract_answer = action_prefix + 'continue turning left to join the main traffic'
                    final_dir_cmd = dir_cmd.left_turn
                if command_int == 2:
                    abstract_answer = action_prefix + 'continue turning right to join the main traffic'
                    final_dir_cmd = dir_cmd.right_turn
                if command_int == 3:
                    abstract_answer = action_prefix + 'continue driving straight to join the main traffic'
                    final_dir_cmd = dir_cmd.straight
        if scenario in ['HighwayCutIn', 'MergerIntoSlowTrafficV2']:
            if command_int == 1:
                abstract_answer = action_prefix + drive_along_prefix + 'then turn left to follow the main traffic'
            if command_int == 2:
                abstract_answer = action_prefix + drive_along_prefix + 'then turn right to follow the main traffic'
            if command_int == 3:
                abstract_answer = action_prefix + drive_along_prefix + 'then go straight to follow the main traffic'
            if ego_vehicle['is_in_junction']:
                if command_int == 1:
                    abstract_answer = action_prefix + 'continue turning left to follow the main traffic'
                    final_dir_cmd = dir_cmd.left_turn
                if command_int == 2:
                    abstract_answer = action_prefix + 'continue turning right to follow the main traffic'
                    final_dir_cmd = dir_cmd.right_turn
                if command_int == 3:
                    abstract_answer = action_prefix + 'continue driving straight to follow the main traffic'
                    final_dir_cmd = dir_cmd.straight
        if scenario in self.leave_highway_scenarios:
            if command_int == 1:
                abstract_answer = action_prefix + drive_along_prefix + 'then turn left to exit the highway'
            if command_int == 2:
                abstract_answer = action_prefix + drive_along_prefix + 'then turn right to exit the highway'
            if command_int == 3:
                abstract_answer = action_prefix + drive_along_prefix + 'then go straight to exit the highway'
            if ego_vehicle['is_in_junction']:
                if command_int == 1:
                    abstract_answer = action_prefix + 'continue turning left to exit the highway'
                    final_dir_cmd = dir_cmd.left_turn
                if command_int == 2:
                    abstract_answer = action_prefix + 'continue turning right to exit the highway'
                    final_dir_cmd = dir_cmd.right_turn
                if command_int == 3:
                    abstract_answer = action_prefix + 'continue driving straight to exit the highway'
                    final_dir_cmd = dir_cmd.straight
    
    abstract_answer = abstract_answer + "."

    new_answer = ""
    if self.answer43_changelane != "" and (not self.answer43_changelane.startswith("No")):
        while(self.answer43_changelane.startswith(" ")):
            self.answer43_changelane = self.answer43_changelane[1:]
        if self.answer43_changelane.startswith("Yes, "):
            self.answer43_changelane = self.answer43_changelane[len("Yes, "):]
            self.answer43_changelane = self.answer43_changelane[0].upper() + self.answer43_changelane[1:]
        new_answer = new_answer + self.answer43_changelane + " "
    if self.answer43_brake != "" and (not self.answer43_brake.startswith("No")) \
        and (not self.answer43_brake.startswith("There is no reason")):
        while(self.answer43_brake.startswith(" ")):
            self.answer43_brake = self.answer43_brake[1:]
        if self.answer43_brake.startswith("Yes, "):
            self.answer43_brake = self.answer43_brake[len("Yes, "):]
            self.answer43_brake = self.answer43_brake[0].upper() + self.answer43_brake[1:]
        new_answer = new_answer + self.answer43_brake + " "
    if new_answer != "":
        abstract_answer = new_answer
        # print(abstract_answer)
    if self.answer43_brake.endswith("already stopped at it.") and final_brake_flag == False and final_stop_flag == False:
        abstract_answer += " The ego vehicle needn't stop at the stop sign now because it has already stopped at it."
    if "continue driving along" in abstract_answer and not ego_vehicle['is_in_junction']:
        abstract_answer += curve_str
    
    if "change lane" in self.answer43_changelane and ego_vehicle['is_in_junction']:
        final_dir_cmd = dir_cmd.follow
    
    if final_lane_change_flag:
        if final_change_dir == 1:
            final_dir_cmd = dir_cmd.right_change
        else:
            final_dir_cmd = dir_cmd.left_change # when both side ok, default left
        
        # if final_dir_cmd == dir_cmd.left_change and 'InvadingTurn' in scenario:
        #     final_dir_cmd = dir_cmd.follow # just deviate
    
    if ego_vehicle['is_in_junction'] and \
       final_dir_cmd in [dir_cmd.left_turn, dir_cmd.right_turn] and \
       'move slowly and wait for a chance' in abstract_answer:
        final_dir_cmd = dir_cmd.straight
        # avoid being hit when crossing an actor flow by the ego vehicle's side

    if final_brake_flag:
        final_spd_cmd = spd_cmd.decelerate
        if ego_vehicle['speed'] < BRAKE_MAX_SPEED:
            final_spd_cmd = spd_cmd.keep
        if ego_vehicle['speed'] < BRAKE_MIN_SPEED:
            final_spd_cmd = spd_cmd.accelerate
    if final_stop_flag:
        final_spd_cmd = spd_cmd.stop
    
    print_debug(f"[debug] after brake and stop, final_spd_cmd = {final_spd_cmd}")

    if self.merging_and_needs_stop:
        final_spd_cmd = spd_cmd.stop
    if ego_vehicle['speed'] < self.ideal_flow_speed and self.merging_and_needs_accelerate:
        final_spd_cmd = spd_cmd.accelerate

    print_debug(f"[debug] after merge, final_spd_cmd = {final_spd_cmd}")
    print_debug(f"[debug] self.ideal_flow_speed = {self.ideal_flow_speed}, self.ideal_flow_speed - DURABLE_SPEED_MARGIN = {self.ideal_flow_speed - DURABLE_SPEED_MARGIN}, NORAML_KEEP_SPEED = {NORMAL_KEEP_SPEED}")
    if final_spd_cmd == spd_cmd.keep and final_brake_flag == False and final_stop_flag == False and \
       ego_vehicle['speed'] < max(self.ideal_flow_speed - DURABLE_SPEED_MARGIN, NORMAL_KEEP_SPEED):
        final_spd_cmd = spd_cmd.accelerate
    if self.opposite_flag: # must accelerate when in the opposite lane
        final_spd_cmd = spd_cmd.accelerate
        if final_lane_change_flag:
            final_dir_cmd = dir_cmd.right_change
    
    print_debug(f"[debug] after keep and opposite, final_spd_cmd = {final_spd_cmd}")

    # out of distribution scenario
    
    nearest_road_wp_loc = get_nearest_waypoint_location(self.map, x=ego_vehicle['location'][0],
                                                                  y=ego_vehicle['location'][1],
                                                                  z=ego_vehicle['location'][2])
    # nearest_road_forward_vec = get_nearest_waypoint_forward_vector(self.map, x=ego_vehicle['location'][0],
    #                                                                 y=ego_vehicle['location'][1],
    #                                                                 z=ego_vehicle['location'][2])
    
    if not ('ParkingExit' in scenario and ego_vehicle['lane_type_str'] == 'Parking') and \
        not is_on_road(self.map, x=ego_vehicle['location'][0],
                                y=ego_vehicle['location'][1],
                                z=ego_vehicle['location'][2]) and \
        not ego_vehicle['is_in_junction']:
        
        if nearest_road_wp_loc is None:
            abstract_answer = "The ego vehicle went out of the road, and it can not find a way back to the road. " +\
                              "It should stop and wait for help." + " " + self.answer43_brake
            final_spd_cmd = spd_cmd.stop
            final_dir_cmd = dir_cmd.straight
        else:
            turn_left = ego_vehicle['lane_deviant_degree'] < 0
            if turn_left:
                abstract_answer = "The ego vehicle went out of the road, and it may turn left " +\
                              "to get back to the road." + " " + self.answer43_brake
                final_spd_cmd = spd_cmd.accelerate if ego_vehicle['speed'] < OOD_ADJUST_SPEED else spd_cmd.keep
                final_dir_cmd = dir_cmd.left_turn
            else:
                abstract_answer = "The ego vehicle went out of the road, and it may turn right " +\
                              "to get back to the road." + " " + self.answer43_brake
                final_spd_cmd = spd_cmd.accelerate if ego_vehicle['speed'] < OOD_ADJUST_SPEED else spd_cmd.keep
                final_dir_cmd = dir_cmd.right_turn
    
    print_debug(f"[debug] after off road, final_spd_cmd = {final_spd_cmd}")
    # print(f"[debug] frame {self.current_measurement_index}, opposite_flag = {self.opposite_flag}, final_dir_cmd = {final_dir_cmd}")

    if not self.opposite_flag and final_dir_cmd == dir_cmd.follow and \
        (not ego_vehicle['is_in_junction']) and \
        nearest_road_wp_loc is not None:

        turn_left = ego_vehicle['lane_deviant_degree'] < 0

        if turn_left:
            turn_degree = abs(ego_vehicle['lane_deviant_degree'])
            deviate_dir = "right"
            turn_dir = "left"
            if turn_degree > ALIGN_THRESHOLD and not self.in_carla: # in carla, will align automatically
                final_dir_cmd = dir_cmd.left_turn
        else:
            turn_degree = abs(ego_vehicle['lane_deviant_degree'])
            deviate_dir = "left"
            turn_dir = "right"
            if turn_degree > ALIGN_THRESHOLD and not self.in_carla: # in carla, will align automatically
                final_dir_cmd = dir_cmd.right_turn
            
        if turn_degree > ALIGN_THRESHOLD:
            abstract_answer = f"The ego vehicle has drifted too far to the {deviate_dir} " +\
                              f"and needs to steer {turn_dir} to align with its current lane." + " " + self.answer43_brake
        if turn_degree > 90:
            abstract_answer = f"The ego vehicle has drifted too far to the reer {deviate_dir} " +\
                              f"and needs to steer {turn_dir} to align with its current lane." + " " + self.answer43_brake
        if turn_degree > 120:
            abstract_answer = "The ego vehicle's direction is almost opposite to the current lane's travel direction " +\
                             f"and needs to make a U-turn to the {turn_dir} to correct it." + " " + self.answer43_brake
        if turn_degree > 150:
            abstract_answer = "The ego vehicle's direction is opposite to the current lane's travel direction " +\
                             f"and needs to make a U-turn to the {turn_dir} to correct it." + " " + self.answer43_brake

    if "deviate slightly" in abstract_answer and 'InvadingTurn' in scenario:
        if final_dir_cmd == dir_cmd.left_change or final_dir_cmd == dir_cmd.right_change: # when on rightmost side, the direction is given left
            final_dir_cmd = dir_cmd.right_deviate
    
    print_debug(f"[debug] after drifted, final_spd_cmd = {final_spd_cmd}")

    if self.in_carla:
        # in reality, this is very unatural.
        # but in carla, if I don't want to miss the deadline to chang lane on highway,
        # the scenario will fail
        # so I have to stop on highway when waiting for a chance to change lane.
        if ('must change lane' in abstract_answer or 'current command orders' in abstract_answer) and \
           final_dir_cmd == dir_cmd.follow and final_change_dir in [1, 2] and \
           (scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios or 'LaneChange' in scenario):
            final_spd_cmd = spd_cmd.stop
            if ego_vehicle['speed'] <= 3.0:
                if final_change_dir == 1:
                    final_dir_cmd = dir_cmd.right_change
                if final_change_dir == 2:
                    final_dir_cmd = dir_cmd.left_change


    self.add_qas_questions(qa_list=qas_conversation_behaviour,
                            qid=43,
                            chain=3,
                            layer=1,
                            qa_type='behaviour',
                            connection_up=-1,
                            connection_down=-1,
                            question=abstract_question,
                            answer=abstract_answer)

    print_debug(f"[debug] finally, final_spd_cmd = {final_spd_cmd}")

    # Update command string if the ego vehicle is already in a junction
    if ego_vehicle['is_in_junction']:
        command_str = command_str.replace('turns', 'continues turning').replace('drives', 'continues driving')
        command_str = command_str.replace('next intersection', 'current intersection')
            
    # # get lane change info
    # now_lane, future_lane, has_changed = detect_future_lane_change_by_time(self.map, ego_vehicle['id'], 
    #                                                             self.current_measurement_path,
    #                                                             k=10)
    # # unify the sign
    # change_left = True
    # if has_changed:
    #     now_lane *= now_lane
    #     future_lane *= future_lane
    #     if future_lane > now_lane:
    #         change_left = False
    
    if not self.in_carla:
        # get future waypoint gt info
        k = 4 * self.frame_rate
        # observe nuScenes evaluation, use 4 second prediction
        measurement_list = {}
        for i in range(0, k + 1):
            measurement_list[i] = get_future_measurements(self.current_measurement_path, i)
        waypoint_list = {}
        for i in range(0, k + 1):
            if measurement_list[i] is None:
                location = get_ego_vehicle_location_from_measurements(current_measurement)
                x = location[0]
                y = location[1]
                z = location[2]
            else:
                location = get_ego_vehicle_location_from_measurements(measurement_list[i])
                x = location[0]
                y = location[1]
                z = location[2]
            future_location = transform_to_ego_coordinates([x, y, z], ego_vehicle['world2ego'])
            waypoint_list[i] = [future_location[0], future_location[1]]
            # print(f"[debug] loc_{i} = {[x, y]}, wp_{i} = {waypoint_list[i]}")

        question = "Provide your planned trajectory for the ego vehicle using waypoints "+\
                    "in the ego vehicle's coordinate system. "+\
                    "Generate one waypoint every 0.5 seconds. "+\
                    "Example output:\n"
                    
        question += '{\n    "0.5s": [x1, y1],\n    "1.0s": [x2, y2],\n    "1.5s": [x3, y3],\n' +\
                        '    "2.0s": [x4, y4],\n    "2.5s": [x5, y5],\n    "3.0s": [x6, y6],\n' +\
                        '    "3.5s": [x7, y7],\n    "4.0s": [x8, y8],\n}'
        
        current_location = transform_to_ego_coordinates(ego_vehicle['location'], ego_vehicle['world2ego'])
        # print(f"[debug] frame {self.current_measurement_index}, cur_loc = {ego_vehicle['location']} -> {current_location}, 0.0s = {[round(x, 2) for x in waypoint_list[int(self.frame_rate * 0)]]}, 0.5s = {[round(x, 2) for x in waypoint_list[int(self.frame_rate * 0.5)]]}, 1.0s = {[round(x, 2) for x in waypoint_list[int(self.frame_rate * 1.0)]]}")
        answer = json.dumps({
            "0.5s": [round(x, 2) for x in waypoint_list[int(self.frame_rate * 0.5)]],
            "1.0s": [round(x, 2) for x in waypoint_list[int(self.frame_rate * 1.0)]],
            "1.5s": [round(x, 2) for x in waypoint_list[int(self.frame_rate * 1.5)]],
            "2.0s": [round(x, 2) for x in waypoint_list[int(self.frame_rate * 2.0)]],
            "2.5s": [round(x, 2) for x in waypoint_list[int(self.frame_rate * 2.5)]],
            "3.0s": [round(x, 2) for x in waypoint_list[int(self.frame_rate * 3.0)]],
            "3.5s": [round(x, 2) for x in waypoint_list[int(self.frame_rate * 3.5)]],
            "4.0s": [round(x, 2) for x in waypoint_list[int(self.frame_rate * 4.0)]],
        })
        self.add_qas_questions(qa_list=qas_conversation_behaviour,
                                    qid=42,
                                    chain=3,
                                    layer=1,
                                    qa_type='behaviour',
                                    connection_up=-1,
                                    connection_down=-1,
                                    question=question,
                                    answer=answer)
    
    ### high-level behaviours
    question = "Provide the appropriate behavior for the ego vehicle, " +\
               "which consists of two keys: the Direction key, which can be " +\
               "'FOLLOW_LANE', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT', 'DEVIATE_LEFT', 'DEVIATE_RIGHT', 'GO_STRAIGHT', 'TURN_LEFT', or 'TURN_RIGHT'; " +\
               "and the Speed key, which can be 'KEEP', 'ACCELERATE', 'DECELERATE', or 'STOP'."
    answer = f"{final_dir_cmd}, {final_spd_cmd}"
    self.add_qas_questions(qa_list=qas_conversation_behaviour,
                                qid=50,
                                chain=3,
                                layer=1,
                                qa_type='behaviour',
                                connection_up=-1,
                                connection_down=-1,
                                question=question,
                                answer=answer)

    ### extra question?

    waiting_for_red_light = 'The ego vehicle should stop because of the traffic light' in abstract_answer
    is_trivial_case = 'The ego vehicle should continue driving along the current lane.' == abstract_answer

    return qas_conversation_behaviour, final_dir_cmd, final_spd_cmd, waiting_for_red_light, is_trivial_case