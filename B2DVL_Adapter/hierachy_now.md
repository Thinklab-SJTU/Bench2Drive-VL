```shell
FUNC generate_perception_questions:
    preprocess data, calculate appendix annotations.

    FUNC generate_vehicle_information:
        for all other vehicles in should_consider_vehicle:
            FUNC determine_other_vehicle_position:
            FUNC determine_vehicle_trajectory:
                USING "steer"
                QUESTION [17] "Where is {other_vehicle_location_description} going?"
            FUNC determine_vehicle_motion_status:
                QUESTION [16] "What is the moving status of {other_vehicle_location_description}?"
            FUNC determine_path_crossing:
                USING "command" "next_command" etc.

    FUNC analyze_road_layout:
        USING "command" "next_command" etc.
        FUNC detect_junction_proximity:
            QUESTION [35] "Is the ego vehicle at a junction?"
        FUNC analyze_important_lanes:
            USING "lane_change" etc.
            QUESTION [36] "The ego vehicle wants to {command_description}. Which lanes are important to watch out for?"
        FUNC analyze_lanes_direction:
            QUESTION [34] "How many lanes are there in the {name} direction {to_or_as} the ego car?"
        FUNC identify_ego_lane:
            QUESTION [33] "On which lane is the ego vehicle (left most lane of the lanes 
            going in the same direction is indicated with 0)?"
        FUNC analyze_lane_marking:
            QUESTION [32] "What lane marking is on the {name} side of the ego car?"
        FUNC analyze_ego_lane_change_direction:
            QUESTION [31] "In which direction is the ego car allowed to change lanes?"
        FUNC lane_change_analysis:
            USING "lane_change" etc.
            QUESTION [30] "From which side are other vehicles allowed to change lanes into the ego lane?"
        QUESTION [44] "Describe the current lane's direction."

    FUNC analyze_environment:
        QUESTION [37] "What is current time and weather?"
        QUESTION [38] "What is current time and weather? What hazards might it bring?"
        QUESTION [39] "What is current time and weather? What should the ego vehicle do according to them?"

        QUESTION [40] "Apart from vehicles on the road, visible pedestrians and the weather, 
        what other factors in the current scenario could pose potential hazards?"
        QUESTION [41] "Apart from vehicles on the road, visible pedestrians and the weather, 
        what other factors in the current scenario could pose potential hazards? 
        What strategies should the ego vehicle adopt to address them?"

    FUNC process_traffic_signs:
        QUESTION [2] "Is the ego vehicle affected by a stop sign?"
        QUESTION [3] "Is the ego vehicle affected by a speed limit sign?"
        QUESTION [4] "List the traffic signs affecting the ego vehicle in the current scenario."

    FUNC process_traffic_lights:
        QUESTION [5] "Is the ego vehicle affected by a traffic light?"
        QUESTION [6] "What is the state of the traffic light?"

    FUNC process_pedestrians:
        QUESTION [1] "How many pedestrians are there?"

    FUNC generate_ego_vehicle_actions:
        FUNC determine_whether_ego_needs_to_change_lanes_due_to_obstruction:
            QUESTION [10] "Does the ego vehicle need to change lanes or deviate from the lane center due to an
            upcoming obstruction?"
            QUESTION [11] "Is there an obstacle on the current road?"
        FUNC determine_whether_ego_needs_to_change_lanes_due_to_other_factor:
            QUESTION [12] "Does the ego vehicle need to change lanes or deviate from the lane 
            for reasons other than the upcoming obstruction? Why?"
        QUESTION [13] "Must the ego vehicle change lane or deviate from the lane now? why?"
        FUNC determine_braking_requirement:
            QUESTION [8] "Does the ego vehicle need to brake? Why?"
        
        for actor in [traffic_light, traffic_sign]:
            FUNC determine_ego_action_based_on_actor:
                QUESTION [9] "What should the ego vehicle do based on the {actor_type}?"
        QUESTION [14] "The list of traffic lights and signs affecting the ego vehicle in current scene is: {sign_list_str}. 
        Based on these traffic signs, what actions should the ego vehicle take respectively?"
        QUESTION [15] "Identify all traffic lights and signs affecting the ego vehicle in current scene. Based on these traffic signs, what actions should the ego vehicle take respectively?"
        
        FUNC add_speed_limit_question:
            QUESTION [7] "What is the current speed limit?"

    QUESTION [18] "What are the important objects in the scene?"
    QUESTION [19] "What are the important objects in the scene? List them from most to least important."

    for vehicle in important_vehicles:
        QUESTION [20] "Where on the road is {vehicle_description} located?"
        QUESTION [21] "What is the rough moving speed and moving direction of {vehicle_description}?"
        QUESTION [22] "What is the exact moving speed and moving direction of {vehicle_description}?"
        QUESTION [23] "The ego vehicle {command_str}. Is {vehicle_location_description} potentially crossing the 
                path of the ego vehicle? If so, why?"
        QUESTION [48] "The ego vehicle {command_str}. Is {vehicle_location_description} potentially crossing the 
                path of the ego vehicle? If so, why? And what action can lead to a collision?"
        QUESTION [49] "The ego vehicle {command_str}. Is {vehicle_location_description} potentially crossing the 
                path of the ego vehicle? If so, what action can lead to a collision?"

    QUESTION [24] "The important vehicles are ..., What is the rough moving speed and moving direction of them?"
    QUESTION [25] "The important vehicles are ..., What is the exact moving speed and moving direction of them?"
    QUESTION [26] "What are the important vehicles and where are they on road?"
    QUESTION [27] "The important vehicles are ..., List their locations on road."
    QUESTION [28] "The important vehicles are ..., Identify potential overlap vehicles and give reasons."
    QUESTION [29] "The important vehicles are ..., List potential overlap vehicles."
    QUESTION [46] "The important vehicles are ..., List potential overlap vehicles and the actions that could lead to a collision."
    QUESTION [47] "The important vehicles are ..., List potential overlap vehicles, overlap reasons and the actions that could lead to a collision."

    QUESTION [42] "Predict the ego vehicle's future waypoint..."
    QUESTION [43] "What is the correct action for the ego vehicle to take now?"
```