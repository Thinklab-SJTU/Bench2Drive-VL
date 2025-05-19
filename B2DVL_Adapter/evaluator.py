import json
from io_utils import *
from math_utils import *
from offline_simulation import *
from prompt_utils import *

SCORE_ROUND_DIGIT = 2
TRAFFIC_SIGN_FUTURE = 10
CHANGE_LANE_FUTURE = 6

def evaluate_question(llm_client, question_dict, anno_path,
                      frame_number, key_object_infos,
                      frame_rate=10, future_data=None, max_tokens=512):
    qid = question_dict['qid']
    Q = question_dict['Q']
    gt = question_dict['actual_gt']
    A = question_dict['VLM_answer']
    while (isinstance(A, list)):
        A = A[0]
    future_answers = future_data
    # print(f"[debug] future answer = {future_answers}")

    current_anno_path = os.path.join(anno_path, f"{frame_number:05d}.json.gz")

    score = 0
    reason = "This answer is completely wrong. (default)"
    if qid in [19]:
        score, reason = important_obj_eval(llm_client, qid, Q, gt, A, key_object_infos, max_tokens)
    elif qid in [7]:
        score, reason = speed_limit_eval(llm_client=llm_client, qid=qid, 
                                         Q=Q, gt=gt, A=A,
                                         frame_number=frame_number,
                                         future_answers=future_answers,
                                         frame_rate=frame_rate,
                                         max_tokens=max_tokens)
    elif qid in [8]:
        score, reason = brake_eval(llm_client=llm_client, qid=qid,
                                   Q=Q, gt=gt, A=A, frame_number=frame_number,
                                   anno_path=anno_path, max_tokens=max_tokens)
    elif qid in [31]:
        score, reason = ego_change_lane_dir(llm_client=llm_client, qid=qid,
                                            Q=Q, gt=gt, A=A, 
                                            max_tokens=max_tokens)
    elif qid in [30]:
        score, reason = other_vehicle_change_lane_dir(llm_client=llm_client, 
                                                      qid=qid,
                                                      Q=Q, gt=gt, A=A, 
                                                      max_tokens=max_tokens)
    elif qid in [2, 3, 4, 15]:
        n = 4
        if qid in [2, 3, 4, 15]: # traffic signs related
            n = TRAFFIC_SIGN_FUTURE
        # if qid in [10, 12, 13]: # change lane
        #     n = CHANGE_LANE_FUTURE
        score, reason = future_acceptable_eval(llm_client=llm_client, qid=qid,
                                               Q=Q, gt=gt, A=A, n=n,
                                               frame_number=frame_number,
                                               future_answers=future_answers,
                                               frame_rate=frame_rate,
                                               max_tokens=max_tokens)
    elif qid in [27, 24, 25, 28, 29, 46, 47]:
        score, reason = listed_eval(llm_client=llm_client,
                                    Q=Q, gt=gt, A=A, qid=qid,
                                    question_dict=question_dict,
                                    key_object_infos=key_object_infos,
                                    max_tokens=max_tokens)
    elif qid in [42]:
        score, reason = simulate_action(llm_client=llm_client,
                                        qid=qid, Q=Q, gt=gt,
                                        A=A, anno_path=anno_path,
                                        frame_number=frame_number,
                                        frame_rate=frame_rate,
                                        max_tokens=max_tokens)
    
    elif qid in [43]:
        score, reason = spd_dir_key_eval_43(A=A, gt_text=gt,
                                            llm_client=llm_client,
                                            max_tokens=max_tokens)
    
    elif qid in [50]:
        score, reason = spd_dir_key_eval_50(A=A, gt_text=gt)

    else:
        score, reason = simple_eval(llm_client=llm_client, qid=qid,
                                    Q=Q, gt=gt, A=A, max_tokens=max_tokens)
    
        
    return round(score, SCORE_ROUND_DIGIT), reason

def get_future_gt_string(frame_number, future_answers=None, frame_rate=10, n=5):
    if future_answers is not None:
        anslen = min(n, len(future_answers))
        OUTSTR = f"Current frame is {frame_number}. "
        OUTSTR += "Here are gts of this question now and in the future: "
        for i in range(0, anslen):
            frame_index = future_answers[i]['frame_number']
            frame_gt = future_answers[i]['qdata'][0]['actual_gt']
            frame_dict = {
                'frame': f"{frame_index} ({(frame_index - frame_number) / frame_rate}s later)",
                'gt': frame_gt
            }
            OUTSTR += json.dumps(frame_dict)
    else: # No future
        OUTSTR = ""
    return OUTSTR

def single_eval(llm_client, qid, system_prompt, Q, gt, A, max_tokens):
    messages = []
    messages.append(
        {
            'role': 'system',
            'content': system_prompt
        }
    )
    messages.append(
        {
            'role': 'user',
            'content': f"question: {Q}\n" +\
                       f"gt: {gt}\n" +\
                       f"answer from VLM: {A}"
        }
    )
    response = llm_client.ask(messages, 
                              max_tokens=max_tokens,
                              response_format={
                                  'type': 'json_object'
                              })
    try:
        json_response = json.loads(response)
        reason_str = json_response.get("reason", json_response.get("reasons", "Parse error!"))
        return json_response['score'], reason_str
    except Exception as e:
        print_error(f'Error when evaluating question {qid}: {Q}, answer: {A}, response: {response}\nError message: {e}')
    return 0, "error"

def single_eval_in_list(llm_client, qid, system_prompt, Q, gt, A, obj_id, max_tokens):
    color_hint_str = check_misinterpret_color(obj_id)
    if color_hint_str != "":
        obj_id = f"{obj_id} ({color_hint_str})"
    messages = []
    messages.append(
        {
            'role': 'system',
            'content': system_prompt
        }
    )
    messages.append(
        {
            'role': 'user',
            'content': f"question: {Q}\n" +\
                       f"gt: {gt}\n" +\
                       f"answer from VLM: {A}" +\
                       f"focus: {obj_id}"
        }
    )
    response = llm_client.ask(messages, 
                              max_tokens=max_tokens,
                              response_format={
                                  'type': 'json_object'
                              })
    try:
        json_response = json.loads(response)
        reason_str = json_response.get("reason", json_response.get("reasons", "Parse error!"))
        return json_response['score'], reason_str
    except Exception as e:
        print_error(f'Error when evaluating question {qid}: {Q}, answer: {A}, response: {response}\nError message: {e}')
    return 0, "error"

def simple_eval(llm_client, qid, Q, gt, A, max_tokens):
    return single_eval(llm_client, qid, SYSTEM_PROMPT + NORMAL_EXAMPLE,
                       Q, gt, A, max_tokens)

def important_obj_eval(llm_client, qid, Q, gt, A, key_object_infos, max_tokens):
    # qid = 19
    # Ground-truth object dictionary parsed from the answer (gt)
    gt_obj_dict = {}
    try:
        gt_obj_dict = parse_objects(gt)
    except:
        print_error(f'Error when evaluating question of important objects, first step, response: {response}')
        return 0, "Error"
    
    # List of object names in ground-truth
    gt_obj_list = [x for x in gt_obj_dict.keys()]
    
    # Prepare messages for LLM request (second stage, to parse objects from VLM answer)
    second_messages = []
    second_messages.append({'role': 'system', 'content': IMPORTANT_OBJECT_PROMPT2 })
    input_json = {"list": gt_obj_list, "sentence": A }
    second_messages.append({'role': 'user', 'content': json.dumps(input_json)})

    # Ask the language model to identify which ground-truth objects are mentioned in the answer
    response = llm_client.ask(second_messages, 
                              max_tokens=max_tokens,
                              response_format={'type': 'json_object'})
    
    vlma_obj_list = []
    try:
        json_response = json.loads(response)
        vlma_obj_list = json_response['list']
    except:
        print_error(f'Error when evaluating question of important objects, first step, response: {response}')
        return 0, "Error"
    
    # print(f"[debug] important object gt = {gt}")
    # print(f"[debug] important object gt_obj_list = {gt_obj_list}")
    # print(f"[debug] important object A = {A}")
    # print(f"[debug] important object vlma_obj_list = {vlma_obj_list}")
    
    # Prepare to compute weights for each ground-truth object
    weight_dict = {}
    total_weight = 0.0
    num_objects = len(gt_obj_list)

    ORDER_RATIO = 5.0 # if equal base weight, first weight is ORDER_RATIO * last weight
    # Ratio of importance between first and last object in the list
    assert(ORDER_RATIO > 1.0)

    min_weight = 100.0 # Track minimum weight to assign small penalty for extra objects
    for i, obj in enumerate(gt_obj_list):
        tag = gt_obj_dict[obj]
        base_weight = 3.0 if tag is not None and tag in key_object_infos and (
            key_object_infos[tag].get('is_role', False) or key_object_infos[tag].get('is_dangerous', False)
        ) else 1.0

        # Higher base weight for important objects (roles or dangerous ones)
        position_weight = num_objects * (ORDER_RATIO) / (ORDER_RATIO - 1.0) - i

        weight_dict[obj] = base_weight * position_weight
        total_weight += weight_dict[obj]
        min_weight = min(weight_dict[obj], min_weight)

    # Normalize weights to sum to 1
    total_weight = max(total_weight, 1.0) # avoid dividing 0
    for obj in weight_dict:
        weight_dict[obj] /= total_weight
    min_weight /= total_weight

    # Only keep model-predicted objects that are actually in ground truth
    included_vlma_obj_list = [x for x in vlma_obj_list if x in gt_obj_list]
    # Compute NDCG score (ranking quality metric)
    ndcg_score = min(ndcg(gt_obj_list, included_vlma_obj_list, weight_dict) * 100.0, 100.0)
    
    EXTRA_RATIO = 2. # Ratio to downscale weights for extra (incorrect) objects
    for obj in vlma_obj_list:
        if obj not in gt_obj_list:
            weight_dict[obj] = min_weight / EXTRA_RATIO
    
    # print(f"[debug] important object weight_dict = {weight_dict}")
    # Compute weighted True Positives, False Positives, and False Negatives
    TP = 0.0
    FP = 0.0
    FN = 0.0
    for obj in vlma_obj_list:
        if obj in gt_obj_dict:
            TP += weight_dict[obj]
        else:
            FP += weight_dict[obj]
    for obj in gt_obj_list:
        if obj not in vlma_obj_list:
            FN += weight_dict[obj]

    # Compute weighted precision, recall, and F1-score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0

    # Final score is NDCG * F1 (both are [0, 100])
    score = ndcg_score * F1
    F1 *= 100.0

    # Compose a JSON-formatted reason string for analysis/debugging
    reason = json.dumps({
        "gt_obj_list": gt_obj_list,
        "vlma_obj_list": vlma_obj_list,
        "weight_dict": weight_dict,
        "ndcg_score": ndcg_score,
        "f1_score": F1
    })
    # print(f"[debug] important object ndcg_score = {ndcg_score}, f1_score = {F1}")

    return score, reason

def speed_limit_eval(llm_client, qid, Q, gt, A, frame_number, future_answers, frame_rate, max_tokens):
    # qid = 7
    APPENDIX = SYSTEM_PROMPT
    if future_answers is not None:
        APPENDIX += FUTURE_PROMPT
    APPENDIX += get_future_gt_string(frame_number, future_answers, frame_rate, TRAFFIC_SIGN_FUTURE)
    APPENDIX += "Moreover, in this question, answering a higher speed limit than gt is a more serious mistake than answering a lower one, because it is more dangerous."
    return single_eval(llm_client, qid, APPENDIX,
                Q, gt, A, max_tokens)

def ego_change_lane_dir(llm_client, qid, Q, gt, A, max_tokens):
    # qid = 31
    APPENDIX = "Moreover, in this task, predicting more lanes than the ground truth is a more severe error compared to predicting fewer lanes, as overprediction can lead to dangerous situations."
    return single_eval(llm_client, qid, SYSTEM_PROMPT + NORMAL_EXAMPLE + APPENDIX,
                       Q, gt, A, max_tokens)

def other_vehicle_change_lane_dir(llm_client, qid, Q, gt, A, max_tokens):
    # qid = 30
    APPENDIX = "Moreover, in this task, predicting fewer lanes than the ground truth is a more severe error compared to predicting more lanes, as underprediction can lead to dangerous situations."
    return single_eval(llm_client, qid, SYSTEM_PROMPT + NORMAL_EXAMPLE + APPENDIX,
                       Q, gt, A, max_tokens)

def brake_eval(llm_client, qid, Q, gt, A, frame_number, anno_path, max_tokens):
    # qid = 8
    current_anno_path = os.path.join(anno_path, f"{frame_number:05d}.json.gz")
    ego_speed = 4.0
    if os.path.exists(current_anno_path):
        current_measurement = load_json_gz(current_anno_path)
        bounding_boxes = current_measurement.get("bounding_boxes", [])
        ego_vehicle = next((item for item in bounding_boxes if item.get("class") == "ego_vehicle"), None)
        ego_speed = ego_vehicle['speed']
    else:
        print_warning(f"{current_anno_path} doe not exist.. this may lead to an inaccurate braking evaluation.")
    APPENDIX = ""
    if ego_speed < 0.5:
        APPENDIX = f"Moreover, given the ego vehicle's low speed ({ego_speed:.2f} m/s), " +\
                    "it is reasonable for the VLM to omit braking, even if the ground truth indicates deceleration or stopping."

    return single_eval(llm_client, qid, SYSTEM_PROMPT + NORMAL_EXAMPLE + APPENDIX,
                       Q, gt, A, max_tokens)

def future_acceptable_eval(llm_client, qid, Q, gt, A, n, frame_number, future_answers, frame_rate, max_tokens):
    # qid = 2, 3, 4, 10, 12, 13, 15
    APPENDIX = SYSTEM_PROMPT
    if future_answers is not None:
        APPENDIX += FUTURE_PROMPT
    APPENDIX += get_future_gt_string(frame_number, future_answers, frame_rate, n)
    return single_eval(llm_client, qid, APPENDIX,
                Q, gt, A, max_tokens)

def extract_keys(text):
    # print(f"[debug] extracting keys from '{text}'")
    direction_keys = {'FOLLOW_LANE', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT', 'DEVIATE_LEFT', 'DEVIATE_RIGHT', 'GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT'}
    speed_keys = {'KEEP', 'ACCELERATE', 'DECELERATE', 'STOP'}
    
    found_keys = set(re.findall(r'\b(FOLLOW_LANE|CHANGE_LANE_LEFT|DEVIATE_LEFT|DEVIATE_RIGHT|CHANGE_LANE_RIGHT|GO_STRAIGHT|TURN_LEFT|TURN_RIGHT|KEEP|ACCELERATE|DECELERATE|STOP)\b', text))
    
    direction = found_keys.intersection(direction_keys)
    speed = found_keys.intersection(speed_keys)

    # print(f"[debug] extraction returned {list(direction)}, {list(speed)}")
    
    return list(direction), list(speed)

def extract_key_list(text):
    direction_keys = {'FOLLOW_LANE', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT',
                      'DEVIATE_LEFT', 'DEVIATE_RIGHT', 'GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT'}
    speed_keys = {'KEEP', 'ACCELERATE', 'DECELERATE', 'STOP'}

    all_keys = re.findall(r'\b(FOLLOW_LANE|CHANGE_LANE_LEFT|DEVIATE_LEFT|DEVIATE_RIGHT|CHANGE_LANE_RIGHT|GO_STRAIGHT|TURN_LEFT|TURN_RIGHT|KEEP|ACCELERATE|DECELERATE|STOP)\b', text)
    
    direction = [k for k in all_keys if k in direction_keys]
    speed = [k for k in all_keys if k in speed_keys]
    
    return direction, speed

def extract_keys_by_llm(llm_client, isA, text, max_tokens):
    messages = []
    if isA:
        SYS_PROMPT = KEY_43_A_PROMPT
    else:
        SYS_PROMPT = KEY_43_GT_PROMPT
    messages.append(
        {
            'role': 'system',
            'content': SYS_PROMPT
        }
    )
    messages.append(
        {
            'role': 'user',
            'content': json.dumps(text)
        }
    )
    response = llm_client.ask(messages, 
                             max_tokens=max_tokens,
                             response_format={
                                 'type': 'json_object'
                             })
    
    try:
        json_response = json.loads(response)
        # print(f"[debug] json_response = {json_response}") #
        if isinstance(json_response, list):
            return json_response
        elif isinstance(json_response, dict):
            direction_key = None
            speed_key = None
            if "direction_key" in json_response:
                direction_key = json_response['direction_key']
            if "Direction_key" in json_response:
                direction_key = json_response['Direction_key']
            if "speed_key" in json_response:
                speed_key = json_response['speed_key']
            if "Speed_key" in json_response:
                speed_key = json_response['Speed_key']
            res_list = []
            if direction_key is not None:
                res_list.append(direction_key)
            if speed_key is not None:
                res_list.append(speed_key)
            return res_list
        else:
            return []
    except Exception as e:
        print_error(f'Error when evaluating question 43: parsing keys in "{text}", response: {response}\nError message: {e}')

def compute_weighted_f1_score(gt, pred):
    speed_penalty = {
        'KEEP': {'ACCELERATE': 1.2, 'DECELERATE': 1.1, 'STOP': 1.0},
        'ACCELERATE': {'KEEP': 1.0, 'DECELERATE': 1.0, 'STOP': 1.0},
        'DECELERATE': {'KEEP': 1.0, 'ACCELERATE': 0.8, 'STOP': 1.0},
        'STOP': {'KEEP': 0.5, 'ACCELERATE': 0.25, 'DECELERATE': 1.0}
    }

    # if gt has GO_STRAIGHT，it's ok to answer 'FOLLOW_LANE'
    # since some intersection has lane in CARLA, 
    # especially highway ones.

    if 'GO_STRAIGHT' in gt:
        pred = ['GO_STRAIGHT' if p == 'FOLLOW_LANE' else p for p in pred]
    
    gt_set = set(gt)
    pred_set = set(pred)
    
    tp = len(gt_set & pred_set)  # True Positive
    fp = len(pred_set - gt_set)  # False Positive
    fn = len(gt_set - pred_set)  # False Negative
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    penalty = 1.0
    for gt_label in gt:
        if gt_label in speed_penalty:
            for pred_label in pred:
                penalty *= speed_penalty[gt_label].get(pred_label, 1.0)
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    weighted_f1 = min(1, f1 * penalty)
    
    return weighted_f1, f1, penalty

def spd_dir_key_eval(gt, pred):
    # qid = 43, 50

    weighted_f1_score, f1, penalty = compute_weighted_f1_score(gt, pred)
    reason_json = {
        "gt": gt,
        "prediction": pred,
        "weighted_f1-score": weighted_f1_score,
        "f1-score": f1,
        "penalty": penalty
    }
    reason_str = json.dumps(reason_json)
    return weighted_f1_score * 100, reason_str

def spd_dir_key_eval_50(A, gt_text):
    # qid = 50
    gt_direction, gt_speed = extract_keys(gt_text)
    gt = gt_direction + gt_speed
    pred_direction, pred_speed = extract_keys(A)
    pred = pred_direction + pred_speed
    return spd_dir_key_eval(gt, pred)

def spd_dir_key_eval_43(A, gt_text, llm_client, max_tokens):
    # qid = 43
    gt = extract_keys_by_llm(llm_client=llm_client,
                             isA=False,
                             text=gt_text,
                             max_tokens=max_tokens)
    pred = extract_keys_by_llm(llm_client=llm_client,
                               isA=True,
                               text=A,
                               max_tokens=max_tokens)
    
    # mentioned lane change, but didn't specify
    if ('CHANGE_LANE_LEFT' in gt or 'CHANGE_LANE_RIGHT' in gt) and ('CHANGE_LANE' in pred):
        original_eval_score, original_eval_reason = spd_dir_key_eval(gt, pred)
        gt_lane_change = None
        if 'CHANGE_LANE_LEFT' in gt:
            gt_lane_change = 'CHANGE_LANE_LEFT'
        else:
            gt_lane_change = 'CHANGE_LANE_RIGHT'
        
        new_pred = []
        for i in range(len(pred)):
            if pred[i] == 'CHANGE_LANE':
                new_pred.append(gt_lane_change)
            else:
                new_pred.append(pred[i])
        new_eval_score, new_eval_reason = spd_dir_key_eval(gt, new_pred)
        final_eval_score = (original_eval_score + new_eval_score) / 2.0
        final_eval_reason = f"The answer didn't specify which lane to change. If wrong, score = {original_eval_score}, reason = {original_eval_reason}; " +\
                            f"if correct, score = {new_eval_score}, reason = {new_eval_reason}. " +\
                            f"So the final score is the average of these two: {final_eval_score}"
        return final_eval_score, final_eval_reason
        
    return spd_dir_key_eval(gt, pred)

def listed_eval(llm_client, qid, Q, gt, A, question_dict, key_object_infos, max_tokens):
    all_reasons = ""
    
    ROLE_WEIGHT = 3.0
    DANGER_WEIGHT = 3.0
    FP_WEIGHT = 1.5 if qid in [28, 29, 46, 47] else 0.25
    # when identifying overlap vehicles, we don't want the VLM to answer too much.

    # Step 1: Parse GT objects
    try:
        gt_obj_dict = parse_objects(gt)
    except:
        print_error(f'Error when evaluating question of listed objects, first step, response: {gt}')
        return 0, "Error"
    gt_obj_list = list(gt_obj_dict.keys())
    

    # Step 2: Use LLM to identify mentioned GT objects in VLM answer
    second_messages = [
        {'role': 'system', 'content': IMPORTANT_OBJECT_PROMPT2},
        {'role': 'user', 'content': json.dumps({"list": gt_obj_list, "sentence": A})}
    ]
    response = llm_client.ask(second_messages, 
                              max_tokens=max_tokens,
                              response_format={'type': 'json_object'})
    try:
        json_response = json.loads(response)
        vlma_obj_list = json_response['list']
    except:
        print_error(f'Error when evaluating question of listed objects, second step, response: {response}')
        return 0, "Error"
    
    # print(f"[debug] gt = {gt}")
    # print(f"[debug] gt_obj_list = {gt_obj_list}")
    # print(f"[debug] A = {A}")
    # print(f"[debug] vlma_obj_list = {gt_obj_list}")

    # Step 3: Build weight dict and per-object score
    weight_dict = {}
    section_score_dict = {}

    for key_obj_id in question_dict['object_tags']:
        weight = 1
        if key_object_infos[key_obj_id].get('is_role', False):
            weight = ROLE_WEIGHT
        elif key_object_infos[key_obj_id].get('is_dangerous', False):
            weight = DANGER_WEIGHT
        
        obj_dscp = key_object_infos[key_obj_id]['Detailed_description']
        obj_str = obj_dscp + key_obj_id
        system_prompt = LISTED_PROMPT + LISTED_EXAMPLE

        single_score, single_reason = single_eval_in_list(
            llm_client=llm_client,
            qid=qid,
            system_prompt=system_prompt,
            Q=Q, gt=gt, A=A,
            obj_id=obj_str,
            max_tokens=max_tokens
        )

        weight_dict[key_obj_id] = weight
        section_score_dict[key_obj_id] = (single_score, single_reason)

    # # Step 4: Normalize weights
    # for obj in weight_dict:
    #     weight_dict[obj] /= total_weight

    # Step 5: Compute weighted TP, FP, FN
    TP = FP = FN = 0.0
    for obj in vlma_obj_list:
        if obj in gt_obj_list:
            if gt_obj_dict[obj] in weight_dict:
                TP += weight_dict[gt_obj_dict[obj]]
        else:
            FP += FP_WEIGHT

    for obj in gt_obj_list:
        if obj not in vlma_obj_list and gt_obj_dict[obj] in weight_dict:
            FN += weight_dict[gt_obj_dict[obj]]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Step 6: Compute weighted total score
    total_weight = 0.0
    for obj_desc in vlma_obj_list:
        if obj_desc in gt_obj_list:
            obj_id = gt_obj_dict[obj_desc]
            total_weight += weight_dict.get(obj_id, FP_WEIGHT)
    total_weight = max(1.0, total_weight) # prevent dividing zero

    raw_score = 0.0
    for obj_desc in vlma_obj_list:
        if obj_desc in gt_obj_list:
            obj_id = gt_obj_dict[obj_desc]
            if obj_id in section_score_dict:
                raw_score += section_score_dict[obj_id][0] * weight_dict.get(obj_id, FP_WEIGHT)
    raw_score /= total_weight

    final_score = min(F1 * raw_score, 100.0)

    # Step 7: Build reason
    object_reason_list = []
    for obj_id in section_score_dict:
        score, reason_text = section_score_dict[obj_id]
        obj_dscp = key_object_infos[obj_id]['Detailed_description']
        weight = weight_dict.get(obj_id, FP_WEIGHT) / total_weight
        object_reason_list.append(
            f"{obj_dscp} ({score:.2f} pt, weight={weight:.2f}): {reason_text}"
        )

    reason = json.dumps({
        "gt_obj_list": gt_obj_list,
        "vlma_obj_list": vlma_obj_list,
        "f1_score": F1 * 100.0,
        "raw_score": raw_score,
        "object_scores": object_reason_list
    }, ensure_ascii=False)

    return final_score, reason

if __name__ == "__main__":
    from mytoken import *
    from api_interface import *
    client = DeepseekInterface(DEEPSEEK_TOKEN, DEEPSEEK_URL)
    gt_text = "GO_STRAIGHT, ACCELERATE"
    vlm_output = """
    Okay, based on the situation and the provided information, here’s the recommended action for the ego vehicle:

    **Action (frame 85):**

    *   **Direction:** DECELERATE
    *   **Speed:** STOP

    **Reasoning:**

    Given the imminent threat of the police car and the wet road conditions, simply “following the road” (FOLLOW_LANE) is completely inappropriate. The priority is to avoid a collision. Decelerating to a stop is the most immediate and safest action.  While a lane change is suggested in the previous response, the immediate danger necessitates bringing the vehicle to a halt.  The lane change can be considered *after* the vehicle has significantly reduced speed and the situation has stabilized (if at all possible).

    Let me know if you’d like me to elaborate on any aspect of this decision!
    """
    print(spd_dir_key_eval_50(A=vlm_output, gt_text=gt_text))

    gt = "There is an accident on the current road. So the ego vehicle must change to the left lane to circumvent the accident. But not now because the target lane is occupied. The ego vehicle should stop and wait for a chance because it must invade the left lane, which is occupied, in order to bypass the accident."
    A = "The ego vehicle should change lanes or deviate from the lane to maintain safety and avoid potential collisions with the faster-moving police cars and trucks. This will allow the ego vehicle to move closer to the faster-moving vehicles, reducing the risk of being overtaken and improving visibility."

    print(spd_dir_key_eval_43(A=A, gt_text=gt, llm_client=client, max_tokens=4096))