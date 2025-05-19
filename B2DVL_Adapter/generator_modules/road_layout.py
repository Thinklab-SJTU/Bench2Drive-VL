from .offline_map_calculations import *
from .graph_utils import *
from .hyper_params import *

def analyze_road_layout(self, ego_vehicle_info, vehicle_info, scene_data, important_objects, key_object_infos, current_measurement, scenario, final_change_dir):
    """
    This method answers the following questions:
    - Is the ego vehicle at a junction?
    - The ego vehicle wants to {command_str}. Which lanes are important to watch out for?
    - How many lanes are there in the {name} direction {to_or_as} the ego car?
    - On which lane is the ego vehicle (left most lane of the lanes going in the same direction is
                                                                                            indicated with 0)?
    - What lane marking is on the {name} side of the ego car?
    - In which direction is the ego car allowed to change lanes?
    - From which side are other vehicles allowed to change lanes into the ego lane?

    Args:
        ego_vehicle_info (dict): A dictionary containing information about the ego vehicle's lane and 
                                                                                        surrounding conditions.
        important_objects (list): A list of important objects around the ego vehicle.
        key_object_infos (dict): A dictionary containing information about key objects around the ego vehicle.
        current_measurement (dict): A dictionary containing the current measurement data.
        scenario (str): The name of the current scenario.

    Returns:
        tuple: A tuple containing the following elements:
            - qas_conversation_roadlayout (list): A list of question-answer pairs related to the road layout.
            - important_objects (list): The updated list of important objects around the ego vehicle.
            - key_object_infos (dict): The updated dictionary containing information about key objects around 
                                                                                                the ego vehicle.
    """

    def lane_change_analysis(is_acceleration_lane, command_int, ego_vehicle_info, is_junction, 
                                                                                qas_conversation_roadlayout):
        """
        Answers "From which side are other vehicles allowed to change lanes into the ego lane?".

        Args:
            is_acceleration_lane (bool): Indicates if the ego vehicle is on an acceleration lane.
            command_int (int): An integer representing a command related to lane changes.
            ego_vehicle_info (dict): A dictionary containing information about the ego vehicle's lane and 
                                                                                            surrounding conditions.
            is_junction (bool): Indicates if the ego vehicle is in a junction.
            qas_conversation_roadlayout (list): A list to store question-answer pairs related to the road layout.
        """

        # Lane change analysis
        question = "From which side are other vehicles allowed to change lanes into the ego lane?"
        if is_acceleration_lane and command_int == 5:
            answer = f"Vehicles could potentially change lanes from the left side but it is very unlikely since " +\
                                                                    f"the ego vehicle is on an acceleration lane."
        elif ego_vehicle_info['lane_change'] == carla.LaneChange.NONE:
            if ego_vehicle_info['num_lanes_same_direction'] == 1:
                answer = "There are no lane changes possible since the ego vehicle is on a one lane road."
            else:
                answer = "There are no lane changes allowed from another driving lane into the ego lane."
        elif ego_vehicle_info['lane_change'] == carla.LaneChange.Right:
            answer = "Vehicles are allowed to change lanes from the right side."
        elif ego_vehicle_info['lane_change'] == carla.LaneChange.Left:
            answer = "Vehicles are allowed to change lanes from the left side."
        elif ego_vehicle_info['lane_change'] == carla.LaneChange.Both:
            answer = "Vehicles are allowed to change lanes from both sides."
        else:
            raise NotImplementedError()

        # Handle parking lanes
        if ego_vehicle_info['parking_left'] and ego_vehicle_info['parking_right'] and \
                                                        ego_vehicle_info['lane_change'] == 0:
            if ego_vehicle_info['num_lanes_opposite_direction'] >= 1:
                answer += " But vehicles that are parked on the right side of the road are allowed to change " +\
                                                                                    f"lanes into the ego lane."
            else:
                answer += " But vehicles that are parked on the left and right side of the road are allowed to " +\
                                                                                f"change lanes into the ego lane."
        elif ego_vehicle_info['parking_left'] and (ego_vehicle_info['lane_change'] != 2 and \
                    ego_vehicle_info['lane_change'] != 3) and ego_vehicle_info['num_lanes_opposite_direction'] == 0:
            if ego_vehicle_info['lane_change'] == 0:
                answer += " But vehicles that are parked on the left side of the road are allowed to change " +\
                                                                                    f"lanes into the ego lane."
            else:
                answer += " And vehicles that are parked on the left side of the road are allowed to change " +\
                                                                                    f"lanes into the ego lane."
        elif ego_vehicle_info['parking_right'] and (ego_vehicle_info['lane_change'] != 1 and \
                                                                            ego_vehicle_info['lane_change'] != 3):
            if ego_vehicle_info['lane_change'] == 0:
                answer += " But vehicles that are parked on the right side of the road are allowed to " +\
                                                                            f"change lanes into the ego lane."
            else:
                answer += " And vehicles that are parked on the right side of the road are allowed to " +\
                                                                            f"change lanes into the ego lane."

        if ego_vehicle_info['lane_type_str'] == 'Parking':
            answer = "The ego vehicle is on a parking lane and vehicles only enter the lane to park."

        # Handle junctions
        if is_junction:
            answer = "It is not possible to tell since the ego vehicle is in a junction."

        # if current_measurement['changed_route'] and ('TwoWays' in scenario or 'HazardAtSideLaneTwoWays' in scenario):
        #     answer = "The ego vehicle overtakes an obstruction. We do not expect vehicles to change " +\
        #                                                                             f"into the ego lane."

        # Store the question-answer pair
        self.all_qa_pairs.append((question, answer))

        # Add the question-answer pair to the conversation roadlayout
        self.add_qas_questions(qa_list=qas_conversation_roadlayout,
                                qid=30,
                                chain=3,
                                layer=6,
                                qa_type='prediction',
                                connection_up=[(6,0)],
                                connection_down=[(3,2), (3,3), (3,4), (3,5)],
                                question=question,
                                answer=answer)

    def analyze_ego_lane_change_direction(is_acceleration_lane, command_int, ego_vehicle_info, vehicle_info, is_junction, 
                                                                                    qas_conversation_roadlayout):
        """
        Answer "In which direction is the ego car allowed to change lanes?".

        Args:
            is_acceleration_lane (bool): Indicates if the ego vehicle is on an acceleration lane.
            command_int (int): An integer representing a command related to lane changes.
            ego_vehicle_info (dict): A dictionary containing information about the ego vehicle's lane and 
                                                                                        surrounding conditions.
            is_junction (bool): Indicates if the ego vehicle is in a junction.
            qas_conversation_roadlayout (list): A list to store question-answer pairs related to the road layout.
        """
        # Lane change direction analysis
        question = "In which direction is the ego car allowed to change lanes?"
        if is_acceleration_lane and command_int == 5:
            answer = f"The ego vehicle is allowed to change lanes to the left to enter the highway."
        elif ego_vehicle_info['lane_change'] == 0:
            if ego_vehicle_info['num_lanes_same_direction'] == 1:
                answer = "The ego vehicle can not change lanes since it is on a one lane road."
            else:
                answer = "The ego vehicle is not allowed to change lanes to another driving lane."
        elif ego_vehicle_info['lane_change'] == 1:
            answer = "The ego vehicle is allowed to change lanes to the right."
        elif ego_vehicle_info['lane_change'] == 2:
            answer = "The ego vehicle is allowed to change lanes to the left."
        elif ego_vehicle_info['lane_change'] == 3:
            answer = "The ego vehicle is allowed to change lanes to the left and right."
        else:
            raise NotImplementedError()
        
        left_clear_distance = get_clear_distance_of_lane(vehicle_info, -1, True)
        right_clear_distance = get_clear_distance_of_lane(vehicle_info, 1, True)
        left_front_clear_distance = get_clear_distance_of_lane(vehicle_info, -1, False)
        right_front_clear_distance = get_clear_distance_of_lane(vehicle_info, 1, False)

        left_occupied = left_clear_distance <= self.lane_clear_threshold and left_front_clear_distance <= self.lane_forward_threshold
        right_occupied = right_clear_distance <= self.lane_clear_threshold and right_front_clear_distance <= self.lane_forward_threshold

        append_sentence = ""
        if ego_vehicle_info['lane_change'] in [1, 3]: # right
            if right_occupied:
                append_sentence = " But it couldn't change to right lane because it is currently occupied."
        if ego_vehicle_info['lane_change'] in [2, 3]: # left
            if left_occupied:
                append_sentence = " But it couldn't change to left lane because it is currently occupied."
        if ego_vehicle_info['lane_change'] in [3]: # both
            if left_occupied and right_occupied:
                append_sentence = " But it couldn't change to either side because they're all currently occupied." 

        answer += append_sentence

        # Handle parking lanes
        if ego_vehicle_info['parking_left'] and ego_vehicle_info['parking_right'] and \
                                                    ego_vehicle_info['lane_change'] == 0:
            if ego_vehicle_info['num_lanes_opposite_direction'] >= 1:
                answer += " But it could change to the parking lane on the right side of the road."
            else:
                answer += " But it could change to the parking lane on the left and right side of the road."
        elif ego_vehicle_info['parking_left'] and (ego_vehicle_info['lane_change'] != 2 and \
                                                    ego_vehicle_info['lane_change'] != 3) and \
                                                    ego_vehicle_info['num_lanes_opposite_direction'] == 0:
            if ego_vehicle_info['lane_change'] == 0:
                answer += " But it could change to the parking lane on the left side of the road."
            else:
                answer += " It could also change to the parking lane on the left side of the road."
        elif ego_vehicle_info['parking_right'] and (ego_vehicle_info['lane_change'] != 1 and \
                                                        ego_vehicle_info['lane_change'] != 3):
            if ego_vehicle_info['lane_change'] == 0:
                answer += " But it could change to the parking lane on the right side of the road."
            else:
                answer += " It could also change to the parking lane on the right side of the road."

        # Handle parking lane
        if ego_vehicle_info['lane_type_str'] == 'Parking':
            answer = "The ego vehicle is on a parking lane and is allowed to merge into the driving lane."

        # Handle junctions
        if is_junction:
            answer = "It is not possible to tell since the ego vehicle is in a junction."

        # if current_measurement['changed_route'] and ('TwoWays' in scenario or 'HazardAtSideLaneTwoWays' in scenario):
        #     answer = "The ego vehicle overtakes an obstruction. It is not expected to change lanes."

        # Store the question-answer pair
        self.all_qa_pairs.append((question, answer))

        # Add the question-answer pair to the conversation roadlayout
        self.add_qas_questions(qa_list=qas_conversation_roadlayout,
                                qid=31,
                                chain=3,
                                layer=5,
                                qa_type='prediction',
                                connection_up=[(6,0)],
                                connection_down=[(3,2),(3,3),(3,4)],
                                question=question,
                                answer=answer)

    def analyze_lane_marking(ego_vehicle_info, qas_conversation_roadlayout):
        """
        Answer "What lane marking is on the {name} side of the ego car?".

        Args:
            ego_vehicle_info (dict): A dictionary containing information about the ego vehicle's lane and 
                                                                                            surrounding conditions.
            qas_conversation_roadlayout (list): A list to store question-answer pairs related to the road layout.
        """

        keys = ['left_lane', 'right_lane']
        names = ['left', 'right']
        for side_key, side_name in zip(keys, names):
            answer = None
            question = f"What lane marking is on the {side_name} side of the ego car?"

            # Determine the lane marking type
            if ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.NONE:
                answer = f"There is no lane marking on the {side_name} side of the ego car."
            elif ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.Broken:
                lanetype = "broken"
            elif ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.Solid:
                lanetype = "solid"
            elif ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.SolidSolid:
                lanetype = "double solid"
            elif ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.Curb:
                lanetype = "curb"
            elif ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.SolidBroken:
                lanetype = "solid broken"
            elif ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.BrokenSolid:
                lanetype = "broken solid"
            elif ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.BrokenBroken:
                lanetype = "broken broken"
            elif ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.Grass:
                lanetype = "grass"
            elif ego_vehicle_info[f'{side_key}_marking_type'] is carla.LaneMarkingType.BottsDots:
                lanetype = "botts dots"
            else:
                raise ValueError(f"Unknown lane marking type {ego_vehicle_info[f'{side_key}_marking_type']}.")

            # Construct the answer string
            if answer is None:
                color = ego_vehicle_info[f'{side_key}_marking_color_str']
                # lower case
                color = color[:1].lower() + color[1:]
                if color == 'other':
                    description_str = f'{lanetype}'
                else:
                    description_str = f'{color} {lanetype} lane'
                answer = f"The lane marking on the {side_name} side of the ego car is a {description_str}."

            # if current_measurement['changed_route'] and \
            #             ('TwoWays' in scenario or 'HazardAtSideLaneTwoWays' in scenario):
                
            #     if side_name == 'right':
            #         color = ego_vehicle_info[f'left_lane_marking_color_str']
            #         # lower case
            #         color = color[:1].lower() + color[1:]
            #         if color == 'other':
            #             description_str = f'{lanetype}'
            #         else:
            #             description_str = f'{color} {lanetype} lane'
            #         answer = f"The lane marking on the right side of the ego car is a {description_str}."
            #     else:
            #         answer = "It is not possible to tell since the ego vehicle overtakes an obstruction."
            

            # Store the question-answer pair
            self.all_qa_pairs.append((question, answer))

            # Add the question-answer pair to the conversation roadlayout
            self.add_qas_questions(qa_list=qas_conversation_roadlayout,
                                    qid=32,
                                    chain=3,
                                    layer=4,
                                    qa_type='perception',
                                    connection_up=[(3,5), (3,6)],
                                    connection_down=[(3,2), (3,3)],
                                    question=question,
                                    answer=answer)

    def identify_ego_lane(ego, is_junction, qas_conversation_roadlayout):
        """
        Answers "On which lane is the ego vehicle (left most lane of the lanes going in the same direction is 
                                                                                                indicated with 0)?".

        Args:
            ego_vehicle_info (dict): A dictionary containing information about the ego vehicle's lane and 
                                                                                            surrounding conditions.
            is_junction (bool): Indicates if the ego vehicle is in a junction.
            qas_conversation_roadlayout (list): A list to store question-answer pairs related to the road layout.
        """

        question = "On which lane is the ego vehicle (left most lane of the lanes going in the same direction "\
                                                                                            "is indicated with 0)?"
        answer = f"The ego vehicle is on lane {ego['ego_lane_number']}."

        # Check if the ego vehicle is on a parking lane
        if ego['lane_type_str'] == 'Parking':
            answer = f"The ego vehicle is on lane {ego['ego_lane_number']} which is the parking lane."

        # Handle junctions
        if is_junction:
            answer = "It is not possible to tell since the ego vehicle is in a junction."

        # if current_measurement['changed_route'] and ('TwoWays' in scenario or 'HazardAtSideLaneTwoWays' in scenario):
        #     answer = f"The ego vehicle is on lane {ego['ego_lane_number']+1} since it overtakes an obstruction."

        # Add the question-answer pair to the conversation roadlayout
        self.add_qas_questions(qa_list=qas_conversation_roadlayout,
                                qid=33,
                                chain=3,
                                layer=3,
                                qa_type='perception',
                                connection_up=[(3,1), (3,4), (3,5), (3,6), (4,0)],
                                connection_down=[(3,0)],
                                question=question,
                                answer=answer)

    def analyze_lanes_direction(ego_vehicle_info, is_junction, qas_conversation_roadlayout):
        """
        Answer "How many lanes are there in the {name} direction {to_or_as} the ego car?".

        Args:
            ego_vehicle_info (dict): A dictionary containing information about the ego vehicle's lane and 
                                                                                    surrounding conditions.
            is_junction (bool): Indicates if the ego vehicle is in a junction.
            qas_conversation_roadlayout (list): A list to store question-answer pairs related to the road layout.
        """

        keys = ['num_lanes_same_direction', 'num_lanes_opposite_direction']
        names = ['same', 'opposite']
        for direction_key, direction_name in zip(keys, names):
            lane_count = number_to_word(ego_vehicle_info[direction_key])
            lane_count_int = ego_vehicle_info[direction_key]

            # Handle parking lanes
            if ego_vehicle_info['lane_type_str'] == 'Parking' and direction_name == 'same':
                lane_count = number_to_word(ego_vehicle_info[direction_key]-1)
                lane_count_int = ego_vehicle_info[direction_key]-1

            s_or_no_s = 's' if lane_count_int > 1 else ''
            are_or_is = 'are' if lane_count_int > 1 else 'is'
            to_or_as = 'to' if direction_name == 'opposite' else 'as'

            question = f"How many lanes are there in the {direction_name} direction {to_or_as} the ego car?"

            if lane_count_int == 0:
                answer = f"There are no lanes in the {direction_name} direction."
            else:
                answer = f"There {are_or_is} {lane_count} lane{s_or_no_s} in the {direction_name} direction."

            # Handle junctions
            if is_junction:
                answer = "It is not possible to tell since the ego vehicle is in a junction."

            # Store the question-answer pair
            self.all_qa_pairs.append((question, answer))

            # Add the question-answer pair to the conversation roadlayout
            self.add_qas_questions(qa_list=qas_conversation_roadlayout,
                                    qid=34,
                                    chain=3,
                                    layer=2,
                                    qa_type='perception',
                                    connection_up=[(3,1), (3,3), (3,4), (3,5), (3,6), (3,7), (4,0)],
                                    connection_down=[(3,0)],
                                    question=question,
                                    answer=answer)
    
    def detect_junction_proximity(is_acceleration_lane, important_objects, key_object_infos, 
                                    is_other_acceleration_lane, is_exit_lane, about_to_exit, ego_vehicle_info, 
                                    is_highway, distance_to_junction, scenario, current_measurement, 
                                    qas_conversation_roadlayout):
        """
        Answer "Is the ego vehicle at a junction?".

        Args:
            is_acceleration_lane (bool): Indicates if the ego vehicle is on an acceleration lane.
            important_objects (list): A list to store important objects detected in the scene.
            key_object_infos (dict): A dictionary to store information about key objects detected.
            is_other_acceleration_lane (bool): Indicates if the ego vehicle is close to an entry lane 
                                                                                                    on the highway.
            is_exit_lane (bool): Indicates if the ego vehicle is on an exit lane.
            about_to_exit (bool): Indicates if the ego vehicle is about to exit the highway.
            ego_vehicle_info (dict): A dictionary containing information about the ego vehicle's lane and 
                                                                                            surrounding conditions.
            is_highway (bool): Indicates if the ego vehicle is on a highway.
            distance_to_junction (float): The distance to the nearest junction.
            scenario (str): The name of the current scenario.
            current_measurement (dict): A dictionary containing the current measurement data.
            qas_conversation_roadlayout (list): A list to store question-answer pairs related to the road layout.
        """
        
        scenario = scenario.split("_")[0]
        question = 'Is the ego vehicle at a junction?'

        if is_acceleration_lane:
            is_junction = False
            answer = 'The ego vehicle is on an acceleration lane and about to enter the highway.'
            important_objects.append('a highway entry')

            key, value = self.generate_object_key_value(
                id=None,
                category='Traffic element', 
                visual_description='Junction',
                detailed_description='the junction',
                object_count=len(key_object_infos),
                obj_dict={'distance': distance_to_junction}
            )
            key_object_infos[key] = value
        elif is_other_acceleration_lane:
            is_junction = False
            answer = "The ego vehicle is on the highway close to the entry lane."
        elif is_exit_lane:
            is_junction = False
            answer = 'The ego vehicle is on an exit lane and about to exit the highway.'
            important_objects.append('a highway exit')

            key, value = self.generate_object_key_value(
                id=None,
                category='Traffic element', 
                visual_description='Junction',
                detailed_description='the junction',
                object_count=len(key_object_infos),
                obj_dict={'distance': distance_to_junction}
            )
            key_object_infos[key] = value
        elif about_to_exit:
            is_junction = False
            answer = "The ego vehicle is on the highway close to the exit lane."
        elif is_highway and ego_vehicle_info['is_in_junction']:
            is_junction = False
            answer = 'The ego vehicle is on the highway potentially close to a junction.'
            important_objects.append('a junction')

            key, value = self.generate_object_key_value(
                id=None,
                category='Traffic element', 
                visual_description='Junction',
                detailed_description='the junction',
                object_count=len(key_object_infos),
                obj_dict={'distance': distance_to_junction}
            )
            key_object_infos[key] = value
        elif ego_vehicle_info['is_in_junction']:
            is_junction = True
            answer = 'The ego vehicle is in a junction.'
            important_objects.append('a junction')

            key, value = self.generate_object_key_value(
                id=None,
                category='Traffic element', 
                visual_description='Junction',
                detailed_description='the junction',
                object_count=len(key_object_infos),
                obj_dict={'distance': distance_to_junction}
            )
            key_object_infos[key] = value
        elif distance_to_junction < 25:
            is_junction = False
            answer = 'The ego vehicle is right before a junction.'
            important_objects.append('a junction')

            key, value = self.generate_object_key_value(
                id=None,
                category='Traffic element', 
                visual_description='Junction',
                detailed_description='the junction',
                object_count=len(key_object_infos),
                obj_dict={'distance': distance_to_junction}
            )
            key_object_infos[key] = value
        else:
            is_junction = False
            answer = 'No, the ego vehicle is not at a junction.'
        
        # Handle specific scenarios
        if scenario == 'InterurbanActorFlow':
            if current_measurement['command'] == 5 and current_measurement['next_command'] != 1:
                answer = "The ego vehicle is on an interurban road close to a point where a new turning " +\
                                                                                            f"lane emerges."
            elif current_measurement['command'] == 5 and current_measurement['next_command'] == 1 and \
                                                                                        distance_to_junction < 35:
                answer = "The ego vehicle is on a turning lane close to a junction."
            elif current_measurement['command'] == 5 and current_measurement['next_command'] == 1:
                answer = "The ego vehicle is on a turning lane approaching a junction."

        # Add the question-answer pair to the conversation roadlayout
        self.add_qas_questions(qa_list=qas_conversation_roadlayout,
                                qid=35,
                                chain=3,
                                layer=0,
                                qa_type='perception',
                                connection_up=[(1,0), (2,0), (3,1), (3,2), (3,3), (4,0)],
                                connection_down=-1,
                                question=question,
                                answer=answer)
        
        return is_junction

    def analyze_important_lanes(command_description, command_int, lane_change_soon, is_junction, ego_vehicle_info, 
                                next_command_int, is_acceleration_lane, about_to_exit, about_to_exit_far, scenario, 
                                current_measurement, qas_conversation_roadlayout, is_highway, 
                                is_other_acceleration_lane, in_junction, final_change_dir):
        """
        Answer "The ego vehicle wants to {command_description}. Which lanes are important to watch out for?".

        Args:
            command_description (str): A string describing the ego vehicle's current command (e.g., "turn left", 
                                                                                                    "go straight").
            command_int (int): An integer representing the current command.
            lane_change_soon (bool): Indicates if the ego vehicle will change lanes soon.
            is_junction (bool): Indicates if the ego vehicle is at a junction.
            ego_vehicle_info (dict): A dictionary containing information about the ego vehicle's lane and 
                                                                                        surrounding conditions.
            next_command_int (int): An integer representing the next command after the current one.
            is_acceleration_lane (bool): Indicates if the ego vehicle is on an acceleration lane.
            about_to_exit (bool): Indicates if the ego vehicle is about to exit the highway.
            about_to_exit_far (bool): Indicates if the ego vehicle is far from the exit lane.
            scenario (str): The name of the current scenario.
            current_measurement (dict): A dictionary containing the current measurement data.
            qas_conversation_roadlayout (list): A list to store question-answer pairs related to the road layout.
        """

        question = f"The ego vehicle wants to {command_description}. Apart from its lane, which lanes are important to watch out for?"
        answer = ''
        scenario = scenario.split("_")[0]
        command_lane_change_flag = False

        if command_int == 1:
            answer = f"The ego vehicle should pay particular attention to traffic coming from the left side of " +\
                            "the intersection and is going straight or turning left, traffic coming from the " +\
                            "right and going straight or turning left and to oncoming traffic."
        elif command_int == 2:
            answer = f"The ego vehicle should pay particular attention to traffic coming straight ahead from " +\
                            f"the left side of the intersection and to oncoming traffic turning left."
        elif command_int == 3:
            if scenario in self.enter_highway_scenarios or scenario in self.leave_highway_scenarios:
                if is_junction:
                    answer = f"The ego vehicle should pay attention to other vehicles in the junction."
                    if scenario in self.enter_highway_scenarios:
                        answer = f"The ego vehicle should pay attention to other vehicles near the highway merging area."
                    if scenario in self.leave_highway_scenarios:
                        answer = f"The ego vehicle should pay attention to other vehicles near the highway exit area."
                    if scenario == "MergerIntoSlowTrafficV2":
                        answer = f"The ego vehicle should pay attention to other vehicles near the highway intersection."
                elif ego_vehicle_info['lane_change'] == carla.LaneChange.NONE:
                    if is_other_acceleration_lane:
                        answer = f"The ego vehicle should pay particular attention to the vehicle on the " \
                                    "acceleration lane to the right."
                    else:
                        answer = f"Since there are no lane changes allowed, the ego does not need to pay " \
                                    "particular attention to any neighboring lane."
                else:
                    if ego_vehicle_info['lane_change'] == carla.LaneChange.Right:
                        add_to_answer = "to the right lane of the highway."
                    elif ego_vehicle_info['lane_change'] == carla.LaneChange.Left:
                        add_to_answer = "to the left lane of the highway."
                    elif ego_vehicle_info['lane_change'] == carla.LaneChange.Both:
                        add_to_answer = "to both neighboring lanes of the highway."

                    if is_other_acceleration_lane:
                        answer = f"The ego vehicle should pay particular attention to the vehicle on the " \
                                    "acceleration lane to the right and " + add_to_answer
                    else:
                        answer = f"The ego vehicle should pay particular attention to " + add_to_answer
            else:
                answer = f"The ego vehicle should pay particular attention to traffic coming from the left " +\
                                f"side of the intersection and is going straight or turning left, traffic " +\
                                f"coming from the right and going straight or turning right and to oncoming " \
                                f"traffic turning left."
        elif command_int == 4 and not lane_change_soon:
            if is_junction:
                answer = f"The ego vehicle should pay attention to other vehicles in the junction."
                if scenario in self.enter_highway_scenarios:
                    answer = f"The ego vehicle should pay attention to other vehicles near the highway merging area."
                if scenario in self.leave_highway_scenarios:
                    answer = f"The ego vehicle should pay attention to other vehicles near the highway exit area."
                if scenario == "MergerIntoSlowTrafficV2":
                    answer = f"The ego vehicle should pay attention to other vehicles near the highway intersection."
            elif ego_vehicle_info['num_lanes_same_direction'] == 1 and \
                                                            ego_vehicle_info['num_lanes_opposite_direction'] == 0:
                if ego_vehicle_info['parking_left'] or ego_vehicle_info['parking_right']:
                    answer = f"There are no other driving lanes to watch out for since the ego vehicle is on a " +\
                                    f"one lane road. But the ego vehicle should watch out for the parking lane."
                else:
                    answer = f"There are no other driving lanes to watch out for since the ego vehicle is on " +\
                                                                                            f"a one lane road."
            elif ego_vehicle_info['num_lanes_same_direction'] == 1 and \
                                                            ego_vehicle_info['num_lanes_opposite_direction'] >= 1:
                if ego_vehicle_info['parking_left'] or ego_vehicle_info['parking_right']:
                    answer = f"The ego vehicle should watch out for traffic in its lane and coming from the oncoming lane and " +\
                                                                                        f"for the parking lane."
                else:
                    answer = f"The ego vehicle should watch out for traffic in its lane and coming from the oncoming lane."
            elif ego_vehicle_info['num_lanes_same_direction'] > 1 and \
                                                            ego_vehicle_info['num_lanes_opposite_direction'] == 0:
                answer = f"The ego vehicle should pay particular attention to traffic changing lanes " +\
                                                                                f"from neighboring lanes."
            elif ego_vehicle_info['num_lanes_same_direction'] > 1 and \
                                                            ego_vehicle_info['num_lanes_opposite_direction'] >= 1:
                if ego_vehicle_info['ego_lane_number'] == 0:
                    # ego driving at left most lane so uncoming traffic is important to watch
                    answer = f"The ego vehicle should pay particular attention to traffic changing lanes from " +\
                                            f"neighboring lanes and for traffic in its lane and coming from the oncoming lane."
                else:
                    # ego not driving at left most lane so uncoming traffic is not important to watch
                    answer = f"The ego vehicle should pay particular attention to traffic changing lanes from " +\
                                                                                            f"neighboring lanes."
            else:
                raise ValueError(f"Unknown number of lanes {ego_vehicle_info['num_lanes_same_direction']} and " +\
                                                        f"{ego_vehicle_info['num_lanes_opposite_direction']}.")
        elif command_int == 5 or (next_command_int == 5 and lane_change_soon):
            # if lane_change_soon:
            #     answer = f"The ego vehicle should pay particular attention to traffic in the left-hand lane and " +\
            #                     f"position itself so that no vehicle is driving on the same height to prepare " +\
            #                     f"for the lane change."
            # else:
            answer = f"The ego vehicle should pay particular attention to traffic in the left-hand lane and " +\
                                                                            f"wait for a gap to change lanes."
            command_lane_change_flag = True
        elif command_int == 6 or (next_command_int == 6 and lane_change_soon):
            # if lane_change_soon:
            #     answer = f"The ego vehicle should pay particular attention to traffic in the right-hand lane " +\
            #                     f"and position itself so that no vehicle is driving on the same height to " +\
            #                     f"prepare for the lane change."
            # else:
            answer = f"The ego vehicle should pay particular attention to traffic in the right-hand lane " +\
                                                                        f"and wait for a gap to change lanes."
            command_lane_change_flag = True
        if 'CrossingBicycleFlow' in scenario and \
            (ego_vehicle_info['is_in_junction'] or ego_vehicle_info['distance_to_junction'] < 15.0):
            answer += " Additionally, the ego vehicle should have an eye on the bike lane oncoming and the one on the right side."
        elif ego_vehicle_info['bike_lane_left'] and ego_vehicle_info['num_lanes_opposite_direction'] == 0 and \
                                                                        ego_vehicle_info['ego_lane_number'] == 0:
            # no oncoming traffic and ego is on left most lane
            answer += " Additionally, the ego vehicle should have an eye on the bike lane on the left side."
        elif ego_vehicle_info['bike_lane_right']:
            answer += " Additionally, the ego vehicle should have an eye on the bike lane on the right side."

        if is_acceleration_lane and command_int==5:
            answer = f"The ego vehicle should pay particular attention to traffic on the rightmost lane of the " +\
                            f"highway, adjust its speed, and position itself so that no vehicle is driving on " +\
                            f"the same height to prepare for the lane change."
        elif is_acceleration_lane and command_int==6:
            raise ValueError("Lane change to the right on acceleration lane is not possible.")
        elif is_acceleration_lane:
            answer = "The ego vehicle should pay particular attention to the traffic on the highway, which is " +\
                                                                            f"close to the acceleration lane."
        elif about_to_exit:
            answer = "The ego vehicle should pay particular attention to the traffic on the exit lane, since " +\
                                                                                        f"they might slow down."
        elif about_to_exit_far:
            answer = "The ego vehicle is still far away from the exit lane, so it should pay attention to the " +\
                                                                                        f"traffic on the highway."

        if scenario in ['HighwayExit', 'MergerIntoSlowTraffic', 'InterurbanActorFlow']:
            if self.first_lane_command == 5:
                if not in_junction:
                    if current_measurement['command'] == 5 and current_measurement['next_command'] not in [1, 3]:
                        answer = "The ego vehicle should pay particular attention to the traffic on the turning lane, " +\
                                                                                            f"since they just exited the highway and might slow down."
                    elif current_measurement['command'] == 5 and current_measurement['next_command'] in [1, 3]:
                        answer = "The ego vehicle should pay particular attention to the traffic on the turning lane as " +\
                                        f"they might slow down and to oncoming traffic the ego vehicle needs to cross " +\
                                        f"in order to enter the left-hand side exit."
                    elif current_measurement['command'] in [1, 3]:
                        answer = "The ego vehicle should pay particular attention to oncoming traffic the ego vehicle " +\
                                        f"needs to cross in order to enter the left-hand side exit."
                else:
                    answer = "The ego vehicle should pay particular attention to the traffic in its lane, since they just exited the highway and might slow down."
            if self.first_lane_command == 6:
                if not in_junction:
                    if current_measurement['command'] == 6 and current_measurement['next_command'] not in [2, 3]:
                        answer = "The ego vehicle should pay particular attention to the traffic on the turning lane, " +\
                                                                                            f"since they just exited the highway and might slow down."
                    elif current_measurement['command'] == 6 and current_measurement['next_command'] in [2, 3]:
                        answer = "The ego vehicle should pay particular attention to the traffic on the turning lane as " +\
                                        f"they might slow down and to oncoming traffic the ego vehicle needs to cross " +\
                                        f"in order to enter the right-hand side exit."
                    elif current_measurement['command'] in [2, 3]:
                        answer = "The ego vehicle should pay particular attention to oncoming traffic the ego vehicle " +\
                                        f"needs to cross in order to enter the right-hand side exit."
                else:
                    answer = "The ego vehicle should pay particular attention to the traffic in its lane, since they just exited the highway and might slow down."
        
        if scenario == 'InterurbanAdvancedActorFlow':
            if current_measurement['command'] in [1, 2, 3] or current_measurement['next_command'] in [1, 2, 3]:
                answer = "The ego vehicle should pay particular attention to the fast traffic on the main road and the traffic on its lane."

        if ego_vehicle_info['lane_type_str'] == 'Parking':
            answer = "The ego vehicle should pay particular attention to the traffic in the lane into which the " +\
                                                            f"ego vehicle wants to enter from the parking space."
        if self.opposite_flag:
            if final_change_dir == 1:
                final_change_dir = 2
            elif final_change_dir == 2:
                final_change_dir = 1
        
        if command_lane_change_flag == False:
            if final_change_dir == 1:
                answer += " The ego vehicle should pay particular attention to traffic in the right-hand lane " +\
                                                                            f"and wait for a gap to change lanes."
            elif final_change_dir == 2:
                answer += " The ego vehicle should pay particular attention to traffic in the left-hand lane " +\
                                                                            f"and wait for a gap to change lanes."
            elif final_change_dir == 3:
                answer += " The ego vehicle should focus on adjacent lanes as it wants to change lane."
        
        if self.opposite_flag:
            answer += " The ego vehicle should pay special attention to oncoming traffic in the same lane, " +\
                "as we are temporarily driving in the wrong direction, which is extremely dangerous!"
        # if current_measurement['changed_route'] and ('TwoWays' in scenario or 'HazardAtSideLaneTwoWays' in scenario):
        #     answer = f"The ego vehicle should keep driving regardless of other vehicles since it overtakes an " +\
        #                                                                                             f"obstruction."

        # Add the question-answer pair to the conversation roadlayout
        self.add_qas_questions(qa_list=qas_conversation_roadlayout,
                                qid=36,
                                chain=3,
                                layer=1,
                                qa_type='prediction',
                                connection_up=[(4,3)],
                                connection_down=[(3,0), (3,2), (3,3)],
                                question=question,
                                answer=answer)


    qas_conversation_roadlayout = []

    distance_to_junction = ego_vehicle_info['distance_to_junction']
    if distance_to_junction is None:
        distance_to_junction = 1000

    speed_limit = self.current_speed_limit

    # Determine if the scenario is a highway scenario and if the ego vehicle is on an acceleration lane
    is_highway = False
    is_acceleration_lane = False
    is_other_acceleration_lane = False
    is_exit_lane = False
    about_to_exit = False
    about_to_exit_far = False
    highway_scenarios = [
        "HighwayCutIn",
        "HighwayExit", 
        "MergerIntoSlowTraffic",
        "MergerIntoSlowTrafficV2",
        "YieldToEmergencyVehicle",
    ]

    if scenario == "HighwayCutIn":
        is_highway = True
        if (ego_vehicle_info['is_in_junction'] or distance_to_junction < 25):
            is_other_acceleration_lane = True
    elif scenario in ["HighwayExit", "MergerIntoSlowTrafficV2", "MergerIntoSlowTraffic", "InterurbanActorFlow"]:
        is_highway = True
        if (ego_vehicle_info['is_in_junction'] or distance_to_junction < 25):
            is_exit_lane = True
        if (ego_vehicle_info['num_lanes_same_direction'] - ego_vehicle_info['ego_lane_number']-1 == 0 and \
                                            current_measurement['command']==6 and distance_to_junction < 40) or \
                                            ego_vehicle_info['is_in_junction'] or distance_to_junction < 10:
            about_to_exit = True
        if (ego_vehicle_info['num_lanes_same_direction'] - ego_vehicle_info['ego_lane_number']-1 == 0 and \
                                                                            current_measurement['command']==6):
            about_to_exit_far = True

    elif scenario in highway_scenarios and speed_limit > 50:
        is_highway = True
        if scenario == 'MergerIntoSlowTraffic' and ego_vehicle_info['num_lanes_same_direction'] == 1 and \
                                                            ego_vehicle_info['num_lanes_opposite_direction'] == 1:
            is_acceleration_lane = False
        elif scenario == 'MergerIntoSlowTraffic' and ego_vehicle_info['num_lanes_same_direction'] > 1:
            is_acceleration_lane = False
        elif scenario in ['InterurbanAdvancedActorFlow'] and (ego_vehicle_info['is_in_junction'] or distance_to_junction < 25):
            is_acceleration_lane = True

    is_junction = detect_junction_proximity(is_acceleration_lane, important_objects, key_object_infos, 
                                            is_other_acceleration_lane, is_exit_lane, about_to_exit, 
                                            ego_vehicle_info, is_highway, distance_to_junction, scenario, 
                                            current_measurement, qas_conversation_roadlayout)
    
    command_int = get_command_int_by_current_measurement(current_measurement=current_measurement,
                                                         ego_vehicle=ego_vehicle_info)
    
    command_next_int = current_measurement['next_command']
    lane_change_soon = False
    map_command = {
        1: 'go left at the intersection',
        2: 'go right at the intersection',
        3: 'go straight at the intersection',
        4: 'follow the road',
        5: 'do a lane change to the left',
        6: 'do a lane change to the right',        
    }
    command_str = map_command[command_int]

    if current_measurement['next_command'] == 5 or current_measurement['next_command'] == 6:
        # get distance to current_measurement['target_point_next'], ego is at 0,0
        # print(current_measurement)
        # target_point_next = current_measurement['target_point_next']
        target_point_next = [current_measurement['x_command_far'] - current_measurement['x'], current_measurement['y_command_far'] - current_measurement['y']]
        distance_to_target_point_next = np.sqrt(target_point_next[0]**2 + target_point_next[1]**2)
        if distance_to_target_point_next < 20 and current_measurement['next_command'] == 5:
            command_str = 'do a lane change to the left soon'
            lane_change_soon = True

        elif distance_to_target_point_next < 20 and current_measurement['next_command'] == 6:
            command_str = 'do a lane change to the right soon'
            lane_change_soon = True
    
    # Update command string if the ego vehicle is already in a junction
    in_junction = ego_vehicle_info['is_in_junction']
    if in_junction:
        command_str = command_str.replace('turns', 'continues turning').replace('drives', 'continues driving')
        command_str = command_str.replace('next intersection', 'current intersection')

    if about_to_exit:
        command_str = 'exit the highway'
    elif about_to_exit_far:
        command_str = 'exit the highway'

    # Analyze important lanes based on the current command and road conditions
    analyze_important_lanes(command_str, command_int, lane_change_soon, is_junction, ego_vehicle_info, 
                            command_next_int, is_acceleration_lane, about_to_exit, about_to_exit_far, 
                            scenario, current_measurement, qas_conversation_roadlayout, is_highway, is_other_acceleration_lane, in_junction, final_change_dir)
    
    # Analyze the number of lanes in the same and opposite directions
    analyze_lanes_direction(ego_vehicle_info, is_junction, qas_conversation_roadlayout)

    # Identify the lane the ego vehicle is currently on
    identify_ego_lane(ego_vehicle_info, is_junction, qas_conversation_roadlayout)

    # Analyze the lane markings on the left and right sides of the ego vehicle
    analyze_lane_marking(ego_vehicle_info, qas_conversation_roadlayout)

    # Analyze the directions in which the ego vehicle is allowed to change lanes
    analyze_ego_lane_change_direction(is_acceleration_lane, command_int, ego_vehicle_info, vehicle_info, is_junction, 
                                                                                    qas_conversation_roadlayout)

    # Analyze the directions from which other vehicles are allowed to change lanes into the ego lane
    lane_change_analysis(is_acceleration_lane, command_int, ego_vehicle_info, is_junction, 
                                                                                    qas_conversation_roadlayout)

    lane_curvature = get_lane_curvature(self.map, ego_vehicle_info['location'], CURVATURE_CHECK_DISTANCE, self.opposite_flag)

    curve_question = "Describe the current lane's direction."
    curve_answer = "It's difficult to tell because it's hard to identify the lane."
    if ego_vehicle_info['lane_type_str'] != 'Driving':
        curve_answer = "It's difficult to tell because the ego vehicle is not in a driving lane."
    if lane_curvature == "left":
        curve_answer = "The current lane is curving to the left."
    if lane_curvature == "right":
        curve_answer = "The current lane is curving to the right."
    if lane_curvature == "straight":
        curve_answer = "The current lane is going straight."
    if lane_curvature == "junction":
        curve_answer = "It's hard to tell since the ego vehicle is at a junction." # in many junctions, 

    self.add_qas_questions(qa_list=qas_conversation_roadlayout,
                            qid=44,
                            chain=3,
                            layer=6,
                            qa_type='perception',
                            connection_up=-1,
                            connection_down=-1,
                            question=curve_question,
                            answer=curve_answer)
    
    return qas_conversation_roadlayout, important_objects, key_object_infos