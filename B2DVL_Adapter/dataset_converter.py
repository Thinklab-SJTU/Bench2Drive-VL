from dataset_config import DatasetConfig
from io_utils import *
from image_process import *
from qa_process import *
import os
from tqdm import tqdm
from waypoint_extractor import *
import re
import multiprocessing

WORKER_COUNT = 8
TAIL_SEC = 4
FRAME_RATE = 10
DATASET_NAME = "B2DVL-base-stage5-50"
# DATASET_NAME = "B2DVL-behav"
BEHAV_TOKEN_MODE = ['xy', 'ds'][0]
ANNO_IMG_QID = [
            17, # Where is {other_vehicle_location_description} going?
            16, # What is the moving status of {other_vehicle_location_description}?
            9, # What should the ego vehicle do based on the {actor_type}?
            14, # The list of traffic lights and signs affecting the ego vehicle in current scene is: {sign_list_str}. Based on these traffic signs, what actions should the ego vehicle take respectively?
            20, # Where on the road is {vehicle_description} located?
            21, # What is the rough moving speed and moving direction of {vehicle_description}?
            22, # What is the exact moving speed and moving direction of {vehicle_description}?
            23, # The ego vehicle {command_str}. Is {vehicle_location_description} potentially crossing the path of the ego vehicle?
            24, 25, 26, 27, 28, 29, # The important vehicles are ...
        ]

FILTER_FLAG = True
TRIVIAL_PORTION = 0.05
REDLIGHT_PORTION = 0.1

def clean_tags(raw_str, no_tags):
    if no_tags:
        cleaned_text = re.sub(r'\(\s*<([^<>]*(?:<[^<>]*>)*[^<>]*)>\s*\)', '', raw_str)
        return cleaned_text
    else:
        return raw_str

def process_scenario_chunk(worker_id, valid_scenarios, format_str, config):
    res_list = []
    vqa_dir = config.input_path
    entry_exits = load_json(config.entry_exit_file)

    all_frame_count = 0
    trivial_count = 0
    redlight_count = 0
    all_question_count = 0
    trivial_question_num_dict = {}
    total_question_num_dict = {}

    for scenario in tqdm(valid_scenarios, desc=f"Worker {worker_id}"):
        scenario_path = os.path.join(vqa_dir, scenario)
        if not os.path.isdir(scenario_path) or '_' not in scenario:
            continue

        entry = entry_exits.get(scenario, {}).get('entry', None)
        exit = entry_exits.get(scenario, {}).get('exit', None)

        if entry is None or exit is None:
            json_files = [f for f in os.listdir(scenario_path) if f.endswith('.json')]
            frame_numbers = [int(f.split('.')[0]) for f in json_files]
            if not frame_numbers:
                continue
            if exit is None:
                exit = max(frame_numbers) - TAIL_SEC * FRAME_RATE
            if entry is None:
                entry = min(frame_numbers)

        prev_vqa = None
        scenario_img_dir = os.path.join(config.out_path, 'images', scenario)
        os.makedirs(scenario_img_dir, exist_ok=True)
        print(f"Worker {worker_id} processing {scenario}, entry = {entry}, exit = {exit}")

        for json_file in sorted(os.listdir(scenario_path)):
            if json_file.endswith('.json'):
                frame_string = json_file.split('.')[0]
                frame_number = int(frame_string)

                if entry is not None and frame_number < entry:
                    continue
                if exit is not None and frame_number >= exit:
                    continue

                curr_path = os.path.join(scenario_path, json_file)
                curr_vqa = load_json(curr_path)

                trivial_portion = trivial_count / (all_frame_count + 1)
                redlight_portion = redlight_count / (all_frame_count + 1)

                if FILTER_FLAG and \
                   not is_good_case(curr_vqa, trivial_portion > TRIVIAL_PORTION, redlight_portion > REDLIGHT_PORTION):
                    print_warning(f'[debug] Neglected frame {frame_number} of {scenario}.')
                    continue

                curr_anno = get_anno_path(config.b2d_path, scenario, frame_number)
                raw_img_paths, anno_img_paths = process_dataset_image(curr_vqa['image_paths'], curr_vqa['key_object_infos'], 
                                                                      config.b2d_path, scenario_img_dir, frame_string)

                subid = 0
                for qid in config.CHAIN['ORDER']:
                    if qid in config.CHAIN['VALID']:
                        _, _, alllist = find_question_and_gt_by_id(qid, curr_vqa)
                        for qdict in alllist:
                            if qid not in total_question_num_dict:
                                total_question_num_dict[qid] = 1
                            else:
                                total_question_num_dict[qid] += 1
                            
                            if FILTER_FLAG and answer_is_trivial(qdict['A'], qid):
                                question_trivial_portion = get_trivial_ratio(qid)
                                if qid not in trivial_question_num_dict:
                                    trivial_question_num_dict[qid] = 0
                                actual_trivial_ratio = trivial_question_num_dict[qid] / (total_question_num_dict[qid])
                                if actual_trivial_ratio > question_trivial_portion:
                                    print_warning(f'[debug] Neglected question {qid} of {scenario} with answer {qdict["A"]}.')
                                    continue
                                else:
                                    trivial_question_num_dict[qid] += 1

                            content = find_context_for_question(qid=qid,
                                                                prev_vqa=prev_vqa,
                                                                vqa=curr_vqa,
                                                                prev=config.CHAIN['PREV'],
                                                                order=config.CHAIN['ORDER'],
                                                                inherit=config.CHAIN['INHERIT'])
                            inherit_list, context_list = content

                            new_image_paths = {}
                            if config.do_surround:
                                if qid in ANNO_IMG_QID and not config.no_tags:
                                    new_image_paths = {
                                        'CAM_FRONT_CONCAT': anno_img_paths['CAM_FRONT_CONCAT'],
                                        'CAM_BACK_CONCAT': anno_img_paths['CAM_BACK_CONCAT'],
                                    }
                                else:
                                    new_image_paths = {
                                        'CAM_FRONT_CONCAT': raw_img_paths['CAM_FRONT_CONCAT'],
                                        'CAM_BACK_CONCAT': raw_img_paths['CAM_BACK_CONCAT'],
                                    }
                            else:
                                if qid in ANNO_IMG_QID and not config.no_tags:
                                    new_image_paths = {
                                        'CAM_FRONT': anno_img_paths['CAM_FRONT']
                                    }
                                else:
                                    new_image_paths = {
                                        'CAM_FRONT': raw_img_paths['CAM_FRONT']
                                    }

                            sample_dict = {
                                "id": f"{scenario}_{frame_string}_{qid}_{subid:05d}",
                                "inherit": inherit_list,
                                "context": context_list,
                                "question": special_process(qdict),
                                "images": new_image_paths
                            }
                            
                            if format_str == "middleware":
                                final_dict = generate_middleware_unit(sample_dict, config.no_tags, curr_anno)
                            elif format_str == "sharegpt":
                                final_dict = generate_sharegpt_unit(sample_dict, config.no_tags, curr_anno)
                            elif format_str == "sharegpt-CoT":
                                _, _, final_dict = generate_sharegpt_CoT_unit(sample_dict, config.no_tags, curr_anno)
                            elif format_str == "middleware-CoT":
                                final_dict = generate_middleware_CoT_unit(raw_dict=sample_dict,
                                                                          no_tags=config.no_tags,
                                                                          curr_anno=curr_anno,
                                                                          curr_vqa_path=json_file,
                                                                          question_dict=qdict,
                                                                          frame_number=frame_number)

                            all_question_count += 1
                            res_list.append(final_dict)
                            subid += 1

                if is_frame_trivial(curr_vqa):
                    trivial_count += 1
                if is_frame_red_light(curr_vqa):
                    redlight_count += 1
                all_frame_count += 1

                prev_vqa = curr_vqa
                tmp_file = os.path.join(config.out_path, f"tmp_save_{worker_id}.json")
                write_json(res_list, tmp_file)

                tmp_stat = {
                    'frame_num': all_frame_count,
                    'trivial_count': trivial_count,
                    'redlight_count': redlight_count,
                    'question_num': all_question_count,
                    'scenario_num': len(valid_scenarios)
                }
                tmp_stat_file = os.path.join(config.out_path, f"tmp_stat_{worker_id}.json")
                write_json(tmp_stat, tmp_stat_file)

    print(f"Worker {worker_id} finished, saved to {tmp_file}.")

def merge_results(config, format_str):
    final_res_list = []
    for worker_id in range(WORKER_COUNT):
        tmp_file = os.path.join(config.out_path, f"tmp_save_{worker_id}.json")
        if os.path.exists(tmp_file):
            with open(tmp_file, 'r', encoding='utf-8') as f:
                final_res_list.extend(json.load(f))
            # os.remove(tmp_file)

    final_stat = {}
    for worker_id in range(WORKER_COUNT):
        tmp_file = os.path.join(config.out_path, f"tmp_stat_{worker_id}.json")
        if os.path.exists(tmp_file):
            with open(tmp_file, 'r', encoding='utf-8') as f:
                worker_stat = json.load(f)
            for key in worker_stat.keys():
                if key not in final_stat:
                    final_stat[key] = worker_stat[key]
                else:
                    final_stat[key] += worker_stat[key]
            # os.remove(tmp_file)

    final_json_path = os.path.join(config.out_path, f"{DATASET_NAME}_{BEHAV_TOKEN_MODE}_{format_str}.json")
    write_json(final_res_list, final_json_path)
    final_stat_path = os.path.join(config.out_path, f"{DATASET_NAME}_{BEHAV_TOKEN_MODE}_{format_str}_stat.json")
    write_json(final_stat, final_stat_path)
    print(f"Final dataset saved to {final_json_path}")

def convert_dataset(format_str):
    clean_cache()
    config = DatasetConfig()
    if format_str not in config.formats:
        print_error(f"Unsupported dataset format: {format_str}")
        return
    
    included_scenarios = []
    if config.do_subset:
        included_scenarios = read_file_lines(config.subset_file)

    vqa_dir = config.input_path
    valid_scenarios = []
    
    for scenario in sorted(os.listdir(vqa_dir)):
        scenario_path = os.path.join(vqa_dir, scenario)
        if not os.path.isdir(scenario_path) or '_' not in scenario:
            continue
        if config.do_subset and scenario not in included_scenarios:
            continue
        valid_scenarios.append(scenario)

    chunked_scenarios = [[] for _ in range(WORKER_COUNT)]
    for idx, scenario in enumerate(valid_scenarios):
        chunked_scenarios[idx % WORKER_COUNT].append(scenario)

    pool = multiprocessing.Pool(WORKER_COUNT)
    for worker_id in range(WORKER_COUNT):
        pool.apply_async(process_scenario_chunk, args=(worker_id, chunked_scenarios[worker_id], format_str, config))

    pool.close()
    pool.join()

    merge_results(config, format_str)

def process_dataset_image(img_dict, key_obj_infos, b2d_path, save_dir, frame_str):
    object_str = ""
    object_tags = {}
    for key, value in key_obj_infos.items():
        if value["Category"] == "Vehicle":
            object_str += f"({key})"
    object_tags = parse_label(object_str)

    for key in img_dict.keys():
        new_path = os.path.join(*img_dict[key].split(os.sep)[-4:])
        new_path = os.path.abspath(os.path.join(b2d_path, new_path))
        img_dict[key] = os.path.abspath(get_real_path(new_path))

    raw_img_dict = get_surround_images(
        do_anno=False, img_dict=img_dict, object_tags={}, save_dir=save_dir,
        frame_str=frame_str
    )
    anno_img_dict = get_surround_images(
        do_anno=True, img_dict=img_dict, object_tags=object_tags, save_dir=save_dir,
        frame_str=frame_str
    )

    reserve_keys = ['CAM_FRONT', 'CAM_FRONT_CONCAT', 'CAM_BACK_CONCAT']
    for key, file in raw_img_dict.items():
        if key not in reserve_keys:
            if os.path.exists(file):
                os.remove(file)
    for key, file in anno_img_dict.items():
        if key not in reserve_keys:
            if os.path.exists(file):
                os.remove(file)
    
    return raw_img_dict, anno_img_dict

def process_special_question(qid, Q):
    if qid in [42]:
        # Q += " (The coordinates are in the ego-vehicle's coordinate system, " +\
        #      "where the positive x-axis represents the forward direction, " +\
        #      "and the positive y-axis represents the right direction.)"
        Q = "Please predict the waypoint tokens for the next 4 seconds, " +\
            "with one set every 0.5 seconds, " +\
            "for a total of 8 sets of relative displacements."
    return Q

def process_special_answer(qid, A):
    if qid in [42]:
        wp_json = json.loads(A)
        deltas, xy_tokens, ds_tokens = extract_delta_and_token_from_json(wp_json)
        if BEHAV_TOKEN_MODE == 'xy':
            return ''.join(xy_tokens)
        elif BEHAV_TOKEN_MODE == 'ds':
            return ''.join(ds_tokens)
        else:
            return A
    return A

def get_anno_path(b2d_root, scenairo_name, frame_number):
    anno_path = os.path.abspath(os.path.join(b2d_root, scenairo_name, 'anno', f'{frame_number:05d}.json.gz'))
    # anno_path = get_real_path(anno_path)
    return anno_path

def get_surround_images(do_anno, img_dict, object_tags, save_dir, frame_str, max_size=800, font_size=20):
    anno_str = "_anno"
    if not do_anno:
        object_tags = {} # only annotate in special cases
        anno_str = "_raw"
    
    annotated_images = {}
    for cam_name, img_path in img_dict.items():
        img_path = os.path.join(img_path)
        annotated_images[cam_name] = img_path
        if cam_name.startswith('CAM'):
            img_resized = generate_anno_img(img_path, object_tags, max_size, font_size, cam_name)
            output_name = f"{cam_name}_{frame_str}{anno_str}"
            output_path = os.path.join(save_dir, f"{output_name}.jpg")
            img_resized.save(output_path)
            annotated_images[cam_name] = os.path.abspath(output_path)
    
    front_concat, back_concat = generate_concat_camera_images(annotated_images)
    
    if front_concat is not None:
        front_concat_path = os.path.join(save_dir, f"CAM_FRONT_CONCAT_{frame_str}{anno_str}.jpg")
        front_concat.save(front_concat_path)
        annotated_images["CAM_FRONT_CONCAT"] = os.path.abspath(front_concat_path)
    
    if back_concat is not None:
        back_concat_path = os.path.join(save_dir, f"CAM_BACK_CONCAT_{frame_str}{anno_str}.jpg")
        back_concat.save(back_concat_path)
        annotated_images["CAM_BACK_CONCAT"] = os.path.abspath(back_concat_path)
    
    return annotated_images

def generate_middleware_unit(raw_dict, no_tags, curr_anno):
    context_str = ""
    for qdict in raw_dict["inherit"]:
        context_str += f"Q (previous frame): {process_special_question(qid=qdict['qid'], Q=clean_tags(qdict['Q'], no_tags))}\n"
        context_str += f"A: {process_special_answer(qid=qdict['qid'], A=clean_tags(qdict['A'], no_tags))}\n"
    for qdict in raw_dict["context"]:
        context_str += f"Q: {process_special_question(qid=qdict['qid'], Q=clean_tags(qdict['Q'], no_tags))}\n"
        context_str += f"A: {process_special_answer(qid=qdict['qid'], A=clean_tags(qdict['A'], no_tags))}\n"
    qdict = raw_dict["question"]
    extra_condition = generate_condition(curr_anno, qdict["qid"])
    current_question = {
        "qid": qdict["qid"],
        "Q": extra_condition + process_special_question(qid=qdict["qid"], Q=clean_tags(qdict["Q"], no_tags)),
        "A": process_special_answer(qid=qdict["qid"],
                                    A=clean_tags(qdict["A"], no_tags)),
        "object_tags": qdict["object_tags"],
    }

    final_dict = {
        "id": raw_dict["id"],
        "context": context_str,
        "question": current_question,
        "image": raw_dict["images"]
    }

    return final_dict

def generate_sharegpt_unit(raw_dict, no_tags, curr_anno):
    img_desc = "The two concatenated images below are from " +\
               "all cameras attatched to the ego vehicle on current frame."
    context_str = ""
    context_str += f"{img_desc}\n<image><image>"

    for qdict in raw_dict["inherit"]:
        context_str += f"Q (previous frame): {process_special_question(qid=qdict['qid'], Q=clean_tags(qdict['Q'], no_tags))}\n"
        context_str += f"A: {process_special_answer(qid=qdict['qid'], A=clean_tags(qdict['A'], no_tags))}\n"
    
    for qdict in raw_dict["context"]:
        context_str += f"Q: {process_special_question(qid=qdict['qid'], Q=clean_tags(qdict['Q'], no_tags))}\n"
        context_str += f"A: {process_special_answer(qid=qdict['qid'], A=clean_tags(qdict['A'], no_tags))}\n"
    
    qdict = raw_dict["question"]
    extra_condition = generate_condition(curr_anno, qdict["qid"])
    # print(f"[debug] extra_condition = {extra_condition}")
    context_str += f"Use information above to answer:\n"
    context_str += (extra_condition + process_special_question(qid=qdict["qid"], Q=clean_tags(qdict["Q"], no_tags)))

    images = [raw_dict["images"]["CAM_FRONT_CONCAT"], raw_dict["images"]["CAM_BACK_CONCAT"]]

    final_dict = {
        "messages": [
            {
                "content": context_str,
                "role": "user"
            },
            {
                "content": process_special_answer(qid=qdict["qid"],
                                                               A=clean_tags(qdict["A"], no_tags)),
                "role": "assistant"
            }
        ],
        "images": images
    }

    return final_dict

def generate_sharegpt_CoT_unit(raw_dict, no_tags, curr_anno):
    img_desc = "The two concatenated images below are from " +\
               "all cameras attatched to the ego vehicle on current frame."
    context_str = ""
    context_str += f"{img_desc}\n<image><image>"

    answer_str = ""

    for qdict in raw_dict["inherit"]:
        context_str += f"Q (previous frame): {process_special_question(qid=qdict['qid'], Q=clean_tags(qdict['Q'], no_tags))} "
        context_str += f"A: {process_special_answer(qid=qdict['qid'], A=clean_tags(qdict['A'], no_tags))} "

    for qdict in raw_dict["context"]:
        answer_str += f"Consider: {transfer_question_in_cot(Q=process_special_question(qid=qdict['qid'], Q=clean_tags(qdict['Q'], no_tags)), qid=qdict['qid'])}"
        answer_str += f"Answer: {transfer_answer_in_cot(A=process_special_answer(qid=qdict['qid'], A=clean_tags(qdict['A'], no_tags)), qid=qdict['qid'])}"

    qdict = raw_dict["question"]
    extra_condition = generate_condition(curr_anno, qdict["qid"])
    full_question = extra_condition + process_special_question(qid=qdict["qid"], Q=clean_tags(qdict["Q"], no_tags))
    # final_answer = process_special_answer(qid=qdict["qid"], A=clean_tags(qdict["A"], no_tags))
    final_question = transfer_question_in_cot(process_special_question(qid=qdict["qid"], Q=clean_tags(qdict["Q"], no_tags)), qid=qdict['qid'])
    final_answer = transfer_answer_in_cot(process_special_answer(qid=qdict["qid"], A=clean_tags(qdict["A"], no_tags)), qid=qdict['qid'])[:-1]

    final_prefix = "Consider the final question: "
    if answer_str == "":
        final_prefix = "I can consider the final question directly: "
    answer_str += f"{final_prefix}{final_question}"
    answer_str += f"Final Answer: {final_answer}"

    answer_str = f"<think>{answer_str}</think>"
    
    answer_str += f"<answer>{final_answer}</answer>"
    
    # print(f"[debug] extra_condition = {extra_condition}")
    context_str += f"Use information above to answer:\n"
    context_str += full_question

    images = [raw_dict["images"]["CAM_FRONT_CONCAT"], raw_dict["images"]["CAM_BACK_CONCAT"]]

    final_dict = {
        "messages": [
            {
                "content": context_str,
                "role": "user"
            },
            {
                "content": answer_str,
                "role": "assistant"
            }
        ],
        "images": images
    }

    return context_str, answer_str, final_dict

def generate_middleware_CoT_unit(raw_dict, no_tags, curr_anno, curr_vqa_path, question_dict, frame_number):
    context_str, answer_str, cot_dict = generate_sharegpt_CoT_unit(raw_dict=raw_dict,
                                                                   no_tags=no_tags,
                                                                   curr_anno=curr_anno)
    images = [raw_dict["images"]["CAM_FRONT_CONCAT"], raw_dict["images"]["CAM_BACK_CONCAT"]]
    final_dict = {
        "sharegpt": cot_dict,
        "images": images,
        "Q": context_str,
        "A": answer_str,
        "anno_path": curr_anno,
        "vqa_path": curr_vqa_path,
        "qdict": question_dict,
        "frame_number": frame_number
    }

    return final_dict


def special_process(qdict):
    if qdict['qid'] == 24:
        qdict['Q'] = "What are the rough moving speed and moving direction of the important vehicles?"
    if qdict['qid'] == 27:
        qdict['Q'] = "Where on the road are the important vehicles located?"
    if qdict['qid'] == 28:
        qdict['Q'] = "Among the important vehicles, please identify those that may overlap with the ego vehicle's path and provide reasons for the overlap."
    return qdict

def find_context_for_question(qid, prev_vqa, vqa, prev, order, inherit):
    inherit_list = []
    context_list = []
    if prev_vqa is not None:
        if qid in inherit:
            for q in inherit[qid]:
                _, _, alllist = find_question_and_gt_by_id(q, prev_vqa)
                inherit_list.extend(alllist)
    if qid in prev:
        for q in order: # strict by order
            if q in prev[qid]:
                _, _, alllist = find_question_and_gt_by_id(q, vqa)
                context_list.extend(alllist)
    for qdict in inherit_list:
        qdict = special_process(qdict)
    for qdict in context_list:
        qdict = special_process(qdict)

    return inherit_list, context_list

def find_question_and_gt_by_id(qid, vqa_data):
    """
    Given qid, return lists of question ans gt answers.
    """
    qlist = []
    alist = []
    alllist = []

    vqa_content = vqa_data['QA']
    for categories in vqa_content.values():
        for qdict in categories:
            if 'qid' in qdict and qdict['qid'] == qid:
                qlist.append(qdict['Q'])
                alist.append(qdict['A'])
                alllist.append(qdict)
    
    if len(qlist) == 0:
        print_error(f'Error: Question with qid {qid} not found. Ignored.')

    return qlist, alist, alllist

if __name__ == "__main__":
    convert_dataset("middleware-CoT")