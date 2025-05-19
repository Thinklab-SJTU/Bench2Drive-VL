from .offline_map_calculations import *
from .graph_utils import *
from .hyper_params import *

def analyze_environment(self, ego, other_vehicles, scene_data, measurements, scenario):
    
    qas_conversation_environment = []
    scenario = scenario.split('_')[0]
    hazard_list = []
    plan_list = []
    #### weather ####
    weather = measurements['weather']
    sun_altitude = weather['sun_altitude_angle']
    time_str = "day"
    weather_str = "clear"
    low_visibility_factor = []
    control_hazard_factor = []
    
    in_tunnel = is_in_tunnel(x=ego['location'][0], y=ego['location'][1], 
                    town_name=self.town_name, road_id=ego['road_id'])

    if not in_tunnel:
        if sun_altitude > NOON_SUN_ALTITUDE:
            time_str = "noon"
        if sun_altitude < DUSK_SUN_ALTITUDE:
            time_str = "daytime, near dusk or dawn"
            # in carla preset, the accurate altitude is 15.0
        if sun_altitude < 0.0:
            time_str = "nighttime"
            low_visibility_factor.append("the nighttime")

        weather_str = "clear"

        if weather['cloudiness'] > CLOUDY_THRESHOLD:
            weather_str = "cloudy"

        precipitation = weather['precipitation']
        if precipitation >= HEAVY_RAIN_THRESHOLD:
            weather_str = "raining heavily" # 100.0 in preset
            low_visibility_factor.append("the heavy rain")
            control_hazard_factor.append("the heavy rain")
        elif precipitation >= MODERATE_RAIN_THRESHOLD:
            weather_str = "raining moderately" # 60.0 in preset
            low_visibility_factor.append("the moderate rain")
            control_hazard_factor.append("the moderate rain")
        elif precipitation >= LIGHT_RAIN_THRESHOLD:
            weather_str = "raining lightly" # 30.0 in preset
    
    if weather['fog_density'] >= FOGGY_THRESHOLD:
        if weather_str == "clear":
            weather_str = "foggy"
        else:
            weather_str = f"{weather_str} and foggy"
        low_visibility_factor.append("the fog")
        # InvadingTurn_Town02_Route95_Weather9
    if weather['wetness'] >= WET_THRESHOLD:
        # PedestrianCrossing_Town12_Route1013_Weather25
        if weather_str == "clear":
            weather_str = "wet"
        else:
            weather_str = f"{weather_str} and wet"
        low_visibility_factor.append("the wetness")

    precipitation_deposits = weather['precipitation_deposits']
    if precipitation_deposits >= ROAD_FLOOD_THRESHOLD:
        control_hazard_factor.append("the road flood")
    elif precipitation_deposits >= ROAD_PUDDLE_THRESHOLD:
        control_hazard_factor.append("the puddles on the road")

    question1 = "What is current time and weather?"
    question2 = "What is current time and weather? What hazards might it bring?"
    question3 = "What is current time and weather? What should the ego vehicle do according to them?"

    answer1 = f"It is {time_str}, {weather_str}."
    answer2 = f"It is {time_str}, {weather_str}."
    answer3 = f"It is {time_str}, {weather_str}."

    if in_tunnel:
        tunnel_ans = "It is impossible to infer the current time and weather from visual information " +\
                        "because the ego vehicle is currently inside a tunnel."
        answer1 = tunnel_ans
        answer2 = tunnel_ans
        answer3 = tunnel_ans

    if len(low_visibility_factor) > 0:
        vfac_str = ""
        for i in range(len(low_visibility_factor)):
            if i == 0:
                vfac_str = f'{low_visibility_factor[i][0].upper()}{low_visibility_factor[i][1:]}'
            elif i == len(low_visibility_factor) - 1:
                vfac_str = f'{vfac_str} and {low_visibility_factor[i]}'
            else:
                vfac_str = f'{vfac_str}, {low_visibility_factor[i]}'
            
        answer2 = f"{answer2} {vfac_str} causes low visibility, " +\
            "which may affect perception results."
        answer3 = f"{answer3} {vfac_str} causes low visibility, " +\
            "which may affect perception results, please drive cautiously to handle unexpected situations."
    else:
        answer2 = f"{answer2} There's no hazard caused by time or weather."
        answer3 = f"{answer3} There's no hazard caused by time or weather."
    
    if len(control_hazard_factor) > 0:
        cfac_str = ""
        for i in range(len(control_hazard_factor)):
            if i == 0:
                cfac_str = f'{control_hazard_factor[i][0].upper()}{control_hazard_factor[i][1:]}'
            elif i == len(control_hazard_factor) - 1:
                cfac_str = f'{cfac_str} and {control_hazard_factor[i]}'
            else:
                cfac_str = f'{cfac_str}, {control_hazard_factor[i]}'
            
        answer2 = f"{answer2} {cfac_str} may cause vehicles to skid or even lose control."
        answer3 = f"{answer3} {cfac_str} may cause vehicles to skid or even lose control, so drive carefully, avoid sudden movements."
    
    self.add_qas_questions(qa_list=qas_conversation_environment,
                                qid=37,
                                chain=3,
                                layer=1,
                                qa_type='perception',
                                connection_up=-1,
                                connection_down=-1,
                                question=question1,
                                answer=answer1)
    
    self.add_qas_questions(qa_list=qas_conversation_environment,
                                qid=38,
                                chain=3,
                                layer=1,
                                qa_type='prediction',
                                connection_up=-1,
                                connection_down=-1,
                                question=question2,
                                answer=answer2)

    self.add_qas_questions(qa_list=qas_conversation_environment,
                                qid=39,
                                chain=3,
                                layer=1,
                                qa_type='planning',
                                connection_up=-1,
                                connection_down=-1,
                                question=question3,
                                answer=answer3)
    
    #### other hazard ####

    other_hazard_list = []
    other_plan_list = []

    question1 = "Apart from vehicles on the road, visible pedestrians and the weather, " +\
                "what other factors in the current scenario could pose potential hazards?"
    question2 = question1 + " What strategies should the ego vehicle adopt to address them?"
                

    # lane_change:
    # 0 - neither; 1 - right; 2 - left; 3 - either
    at_leftmost = ego['is_in_junction'] is False and ego['lane_change'] in [0, 1] and ego['num_lanes_opposite_direction'] == 0
    at_rightmost = ego['is_in_junction'] is False and ego['lane_change'] in [0, 2]
    # print(f'[debug] at_leftmost = {at_leftmost}, at_rightmost = {at_rightmost}')
    # print(f"[debug] ego['is_in_junciton'] = {ego['is_in_junction']}, ego['lane_change'] = {ego['lane_change']}, opposite_lane = {ego['num_lanes_opposite_direction']}")

    if precipitation_deposits >= ROAD_FLOOD_THRESHOLD:
        other_hazard_list.append("The flood on the road will make vehicle more difficult to control")
        other_plan_list.append("the ego vehicle should drive cautiously and be prepared for potential control loss")
    elif precipitation_deposits >= ROAD_PUDDLE_THRESHOLD:
        other_hazard_list.append("The puddles on the road will make vehicle more difficult to control")
        other_plan_list.append("the ego vehicle should drive cautiously and be prepared for potential control loss")

    left_parking_count = 0
    right_parking_count = 0
    if at_leftmost:
        for actor in scene_data:
            if actor['class'] == 'vehicle' and actor['state'] == 'static':
                # print(f'[debug] left parking actor = {actor}')
                if 0.0 < actor['position'][0] <= PARKED_VEHICLE_MAX_X and \
                    actor['position'][1] < 0.0:
                        left_parking_count += 1
    if at_rightmost:
        for actor in scene_data:
            if actor['class'] == 'vehicle' and actor['state'] == 'static':
                # print(f'[debug] right parking actor = {actor}')
                if 0.0 < actor['position'][0] <= PARKED_VEHICLE_MAX_X and \
                    actor['position'][1] > 0.0:
                        right_parking_count += 1
    
    # print(f"[debug] left_count = {left_parking_count}, right_count = {right_parking_count}")
    
    if (left_parking_count + right_parking_count) > 0:
        dir_str = "to the side"
        plural = ""
        pronoun = "it"
        pronoun2 = "it"

        if left_parking_count > 0:
            dir_str = "on left side"
        if right_parking_count > 0:
            dir_str = "on right side"
        if left_parking_count > 0 and right_parking_count > 0:
            dir_str = "on both sides"
        if (left_parking_count + right_parking_count) > 1:
            plural = "s"
            pronoun = "they"
            pronoun2 = "them"
                
        other_hazard_list.append(f"The parked vehicle{plural} {dir_str} may suddenly start and invade the ego vehicle's lane. " +\
                                        f"Additionally, {pronoun} can create blind spots")
        other_plan_list.append(f"the ego vehicle should keep an eye on {pronoun2}, staying alert for sudden movements or " +\
                                    "dangerous objects such as pedestrians or bicycles that might emerge from the blind spots")
    
    if scenario in ['ParkingCrossingPedestrian', 'DynamicObjectCrossing']:
        if not self.in_carla and self.first_walker_position is not None:
            relative_position = transform_to_ego_coordinates(self.first_walker_position, ego['world2ego'])
            relative_distance = math.sqrt(math.pow(relative_position[0], 2) + math.pow(relative_position[1], 2))
            if relative_distance < BLIND_SPOT_MAX_DISTANCE and relative_position[0] > BLIND_SPOT_MIN_X:
                on_the_right = relative_position[1] >= 0.0

                pronoun = ""
                if scenario in ['ParkingCrossingPedestrian']:
                    pronoun = "is a sidewalk with a parking vehicle "
                if scenario in ['DynamicObjectCrossing']:
                    pronoun = "is a sidewalk with obstacles "
                if on_the_right:
                    dir_str = "on the right"
                else:
                    dir_str = "on the left"
                    
                other_hazard_list.append(f"There {pronoun} {dir_str}, creating blind spots")
                other_plan_list.append(f"the ego vehicle should be cautious of dangerous objects that might emerge from these blind spots")
        elif self.in_carla and self.role_actor is not None and self.other_info is not None:
            first_walker_location = [self.role_transform.location.x, self.role_transform.location.y, self.role_transform.location.z]
            relative_position = transform_to_ego_coordinates(first_walker_location, ego['world2ego'])
            relative_distance = math.sqrt(math.pow(relative_position[0], 2) + math.pow(relative_position[1], 2))
            if relative_distance < BLIND_SPOT_MAX_DISTANCE and relative_position[0] > BLIND_SPOT_MIN_X:
                on_the_right = self.other_info['direction']['value'] == 'right'

                pronoun = ""
                if scenario in ['ParkingCrossingPedestrian']:
                    pronoun = "is a sidewalk with a parking vehicle "
                if scenario in ['DynamicObjectCrossing']:
                    pronoun = "is a sidewalk with obstacles "
                if on_the_right:
                    dir_str = "on the right"
                else:
                    dir_str = "on the left"
                    
                other_hazard_list.append(f"There {pronoun} {dir_str}, creating blind spots")
                other_plan_list.append(f"the ego vehicle should be cautious of dangerous objects that might emerge from these blind spots")

    scenario = scenario.split('.')[-1]
    if 'ControlLoss' in scenario:
        debris = [x for x in scene_data if 'static.prop.dirtdebris' in x['type_id'] and\
                                            x['lane_id'] == ego['lane_id'] and\
                                            x['position'][0] > 0.0]
        if debris and len(debris) > 0:
            other_hazard_list.append(f"There is dirt debris ahead that may cause the vehicle to skid")
            other_plan_list.append(f"the ego vehicle should drive cautiously and get prepared for potential control loss")

    answer1 = ""
    answer2 = ""
    if len(other_hazard_list) > 0:
        for hazard, plan in zip(other_hazard_list, other_plan_list):
            answer1 = f"{answer1}{hazard}. "
            answer2 = f"{answer2}{hazard}, so {plan}. "
    else:
        answer1 = "There's no serious potential hazard apart from them."
        answer2 = answer1
    
    self.add_qas_questions(qa_list=qas_conversation_environment,
                                qid=40,
                                chain=3,
                                layer=1,
                                qa_type='prediction',
                                connection_up=-1,
                                connection_down=-1,
                                question=question1,
                                answer=answer1)

    self.add_qas_questions(qa_list=qas_conversation_environment,
                                qid=41,
                                chain=3,
                                layer=1,
                                qa_type='planning',
                                connection_up=-1,
                                connection_down=-1,
                                question=question2,
                                answer=answer2)

    return qas_conversation_environment