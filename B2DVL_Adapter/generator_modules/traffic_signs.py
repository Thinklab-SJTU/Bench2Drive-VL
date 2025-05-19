from .offline_map_calculations import *
from .graph_utils import *
from .hyper_params import *
from io_utils import print_debug

def process_traffic_signs(self, traffic_signs, important_objects, object_infos):
    """
    Answers the question:
    Is the ego vehicle affected by a stop sign? (affected by stop sign & distance < 40)
    Is the ego vehicle affected by a speed limit? (affected by stop sign & distance < 40)
    List the traffic signs affecting the ego_vehicle in the current scenario
    """
    
    qas_traffic_sign = []

    traffic_sign_info = {}
    traffic_sign_affects_ego = {}
    traffic_sign_tags = {}
    
    speed_limit_val = INF_MAX

    for sign_key in self.traffic_sign_map.keys():
        traffic_sign_info[sign_key] = []
        traffic_sign_affects_ego[sign_key] = False
        traffic_sign_tags[sign_key] = []

    traffic_signs = sorted(traffic_signs, key=lambda x: x['distance'])
    sign_list_str = ""

    for traffic_sign in traffic_signs:
        # print_debug(f"[debug] processing traffic sign, type_id = {traffic_sign['type_id']}, id = {traffic_sign['id']}")
        for sign_key, sign_value in self.traffic_sign_map.items():
            if sign_value['type_id'] in traffic_sign['type_id'] and \
                ('speed_limit' in sign_value['type_id'] or traffic_sign['affects_ego'] is True) and \
                traffic_sign['distance'] < TRAFFIC_SIGN_CONSIDER_RADIUS:
                # print_debug(f"[debug] considered traffic sign, type_id = {traffic_sign['type_id']}, id = {traffic_sign['id']}")
                if 'speed_limit' in sign_value['type_id'] and \
                    (self.ahead_speed_limit is None or (self.ahead_speed_limit is not None and \
                                                        self.ahead_speed_limit['id'] != traffic_sign['id'])):
                    # only reserve the speed limit ahead
                    continue
                
                if traffic_sign_affects_ego[sign_key] is True:
                    # only one is valid
                    continue

                v_str = sign_value['visual_description']
                d_str = sign_value['detailed_description']
                if 'speed_limit' in sign_value['type_id']:
                    speed_limit_val = traffic_sign['type_id'].split('.')[-1]
                    v_str = f'{speed_limit_val} km/h speed limit sign'
                    d_str = f'the {speed_limit_val} km/h speed limit sign ahead'

                traffic_sign_affects_ego[sign_key] = True
                important_objects.append(d_str)
                # projected_points, projected_points_meters = project_all_corners(traffic_sign, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
                project_dict = get_project_camera_and_corners(traffic_sign, self.CAM_DICT)
                key, value = self.generate_object_key_value(
                    id=traffic_sign['id'],
                    category='Traffic element',
                    visual_description=v_str,
                    detailed_description=d_str,
                    object_count=len(object_infos),
                    obj_dict=traffic_sign,
                    projected_dict=project_dict
                )
                object_infos[key] = value
                traffic_sign_tags[sign_key] = [key]
                traffic_sign_info[sign_key] = traffic_sign

                if sign_list_str == "":
                    sign_list_str = f"{d_str[0].upper()}{d_str[1:]}({key})"
                else:
                    sign_list_str = f"{sign_list_str}, {d_str}({key})"
    
    if self.passed_speed_limit is not None:
        if sign_list_str != "":
            sign_list_str = f"{sign_list_str}, the passed {self.current_speed_limit} km/h speed limit sign"
        else:
            sign_list_str = f"The passed {self.current_speed_limit} km/h speed limit sign"
    if sign_list_str != "":
        sign_list_str = f"{sign_list_str}."
    else:
        sign_list_str = "There's no traffic sign affecting the ego vehicle."

    question = "Is the ego vehicle affected by a stop sign?"
    if traffic_sign_affects_ego['stop_sign']:
        answer = f"Yes, the ego vehicle is affected by a stop sign({traffic_sign_tags['stop_sign']}), which has not been cleared yet."
    else:
        dist_thres = CARLA_STOP_SIGN_DISTANCE_THRESHOLD if self.in_carla else STOP_SIGN_DISTANCE_THRESHOLD
        cleared_stop_signs = [x for x in traffic_signs if x['distance'] < dist_thres 
                                                    and not x['affects_ego'] 
                                                    and x['position'][0] > STOP_SIGN_AHEAD_THRESHOLD
                                                    and x['type_id'] == 'traffic.stop']

        if cleared_stop_signs:
            answer = f"Yes, the ego vehicle was affected by a stop sign({traffic_sign_tags['stop_sign']}), which has already been cleared."
        else:
            answer = "No, the ego vehicle is not affected by a stop sign."

    # Add the question and answer to the conversation
    self.add_qas_questions(qa_list=qas_traffic_sign,
                            qid=2, 
                            chain=1,
                            layer=0,
                            qa_type='prediction',
                            connection_up=[(1, 1)], 
                            connection_down=[(3, 0)], 
                            question=question,
                            answer=answer,
                            object_tags=traffic_sign_tags['stop_sign'])

    question = "Is the ego vehicle affected by a speed limit sign?"
    if self.passed_speed_limit is not None:
        answer = f"Yes, the ego vehicle is affected by the passed {self.current_speed_limit} km/h speed limit sign"
        important_objects.append(f"the passed {self.current_speed_limit} km/h speed limit sign")
        if self.ahead_speed_limit is not None:
            answer = f"{answer}, and the speed limit will soon change to {self.future_speed_limit} km/h because of the speed limit sign ahead({traffic_sign_tags['speed_limit']})."
        else:
            answer = f"{answer}."
    else:
        if self.ahead_speed_limit is not None:
            answer = f"Yes, the speed limit will soon change to {self.future_speed_limit} km/h because of the speed limit sign ahead({traffic_sign_tags['speed_limit']})."
        else:
            answer = "No, the ego vehicle is not affected by a speed limit sign."

    # Add the question and answer to the conversation
    self.add_qas_questions(qa_list=qas_traffic_sign, 
                            qid=3, 
                            chain=1,
                            layer=0,
                            qa_type='prediction',
                            connection_up=[(1, 1)], 
                            connection_down=[(3, 0)], 
                            question=question,
                            answer=answer,
                            object_tags=traffic_sign_tags['speed_limit'])
    
    question = "List the traffic signs affecting the ego vehicle in the current scenario."
    all_tags = []
    for tag_list in traffic_sign_tags.values():
        if tag_list is not None:
            all_tags.extend(tag_list)
    self.add_qas_questions(qa_list=qas_traffic_sign, 
                            qid=4, 
                            chain=1,
                            layer=0,
                            qa_type='prediction',
                            connection_up=[(1, 1)], 
                            connection_down=[(3, 0)], 
                            question=question,
                            answer=sign_list_str,
                            object_tags=all_tags)

    return qas_traffic_sign, important_objects, object_infos, traffic_sign_info, traffic_sign_tags

def process_traffic_lights(self, traffic_lights, ego_vehicle, important_objects, 
                        object_infos):
    """
    Answers the questions:
    Is the ego vehicle affected by a traffic light?
    What is the state of the traffic light?
    """

    qas_traffic_light = []
    traffic_light_info = None
    object_tags = []

    traffic_light_affects_ego = False

    for traffic_light in traffic_lights:
        if traffic_light['affects_ego']: # and ego_vehicle['traffic_light_state'] != 'None':
            if traffic_light['distance'] < 45:
                state = traffic_light['state']
                # for traffic_light
                # 0 - Red; 1 - Yellow; 2 - Green; 3 - Off; 4 - Unknown;
                if state == 2:
                    ego_vehicle['traffic_light_state'] = "green"
                    state = "green"
                elif state == 1:
                    ego_vehicle['traffic_light_state'] = "yellow"
                    state = "yellow"
                elif state == 0:
                    ego_vehicle['traffic_light_state'] = "red"
                    state = "red"
                elif state == 3:
                    state = "off"
                    continue
                else:
                    state = "unknown"
                    continue
                # state = state[:1].lower() + state[1:]
                traffic_light_affects_ego = True
                traffic_light_info = traffic_light
                break

    question = "Is the ego vehicle affected by a traffic light?"
    if traffic_light_affects_ego:
        answer = "Yes, the ego vehicle is affected by a traffic light."
        important_objects.append(f'the {state} traffic light')
        # projected_points, projected_points_meters = project_all_corners(traffic_light, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
        project_dict = get_project_camera_and_corners(traffic_light, self.CAM_DICT)
        visual_description = f'{ego_vehicle["traffic_light_state"]} traffic light'
        key, value = self.generate_object_key_value(id=traffic_light['id'],
                                                    category='Traffic element',
                                                    visual_description=visual_description,
                                                    detailed_description=f"the {visual_description}",
                                                    object_count=len(object_infos),
                                                    obj_dict=traffic_light,
                                                    projected_dict=project_dict)
        object_infos[key] = value
        object_tags = [key]
    else:
        answer = "No, the ego vehicle is not affected by a traffic light."

    # Add the question and answer to the conversation
    self.add_qas_questions(qa_list=qas_traffic_light, 
                        qid=5, 
                        chain=2,
                        layer=0,
                        qa_type='perception',
                        connection_up=[(2, 1)], 
                        connection_down=[(3, 0)], 
                        question=question,
                        answer=answer,
                        object_tags=object_tags)

    # Add the question about the traffic light state
    question = "What is the state of the traffic light?"
    if traffic_light_affects_ego:
        answer = f"The traffic light({object_tags}) is {state}."
    else:
        answer = "There is no traffic light affecting the ego vehicle."

    # Add the question and answer to the conversation
    self.add_qas_questions(qa_list=qas_traffic_light, 
                        qid=6, 
                        chain=2,
                        layer=1,
                        qa_type='prediction',
                        connection_up=[(2, 2)], 
                        connection_down=[(2, 0)], 
                        question=question,
                        answer=answer,
                        object_tags=object_tags)

    return qas_traffic_light, important_objects, object_infos, traffic_light_info, object_tags