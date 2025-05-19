from math_utils import *
from io_utils import *

def check_scene_overlap(ego_location, ego_yaw, ego_extent, measurements, collision_radius, delta_time):
    """
    Calculate the Euclidean distance between two points in a 2D plane (x, y).
        
    Returns:
        list of dict: Collision infos of this scene
    """

    collision_infos = []
    all_rects = []
    bounding_boxes = measurements.get("bounding_boxes", [])
    ego_rect_corners = calculate_rectangle_corners(center_x=ego_location[0],
                                                   center_y=ego_location[1],
                                                   yaw=ego_yaw, 
                                                   extent_x=ego_extent[0]+collision_radius, 
                                                   extent_y=ego_extent[1]+collision_radius)
    # print(f"[debug] ego_rect_corners = {ego_rect_corners}")
    all_rects.append(ego_rect_corners)

    # print(f"[debug] ego_rect_corners = {ego_rect_corners}")
    for obj_dict in bounding_boxes:
        if obj_dict['class'] in ['vehicle', 'walker']:
            obj_location = obj_dict['location']
            obj_distance = obj_dict['distance']
            if obj_distance < 60.0:
                obj_rotation = obj_dict['rotation']
                obj_world_cord = obj_dict['world_cord']
                obj_extent = obj_dict['extent']
                obj_rect_corners = calculate_rectangle_corners(center_x=obj_location[0],
                                                            center_y=obj_location[1],
                                                            yaw=obj_rotation[2],
                                                            extent_x=obj_extent[0]+collision_radius,
                                                            extent_y=obj_extent[1]+collision_radius)
                # print(f"[debug] obj_rect_corners = {obj_rect_corners}")
                # print(f"[debug] obj_rect_corners = {obj_rect_corners}\n        while world_cord = {obj_world_cord}")
                collide_flag, iou = is_intersecting_and_iou(ego_rect_corners, obj_rect_corners)
                if collide_flag:
                    issue_dict = {
                        "obj_dict": obj_dict,
                        "iou": iou,
                        "delta_time": delta_time
                    }
                    collision_infos.append(issue_dict)
                all_rects.append(obj_rect_corners)

    # plot_points_with_random_colors(all_rects) # You can generate image for debugging
            
    return collision_infos

def simulate_action(llm_client, qid, Q, gt, A, anno_path, frame_number, frame_rate, max_tokens):
    
    collision_score = 1.0
    collision_penalty = 0.5
    
    if isinstance(A, list):
        A = A[0]
    A = clean_json_string(A)
    # print(f"answer = {A}")
    wp_dict = {}
    try:
        wp_dict = json.loads(A)
        # print(f"[debug] wp_dict = {wp_dict}")
    except json.JSONDecodeError as e:
        print_error(f"Error decoding JSON: {e}\nWhile JSON content is: \n{A}")
        return 0, "JSON has incorrect format."
    
    current_anno_path = os.path.join(anno_path, f"{frame_number:05d}.json.gz")
    current_measurement = load_json_gz(current_anno_path)
    bounding_boxes = current_measurement.get("bounding_boxes", [])
    ego_vehicle = next((item for item in bounding_boxes if item.get("class") == "ego_vehicle"), None)
    current_inv_mat = ego_vehicle['world2ego']
    current_world_pos = ego_vehicle['location']
    current_rotation = ego_vehicle['rotation']
    ego_extent = ego_vehicle['extent']
    # print(f"[debug] current_rotation = {current_rotation}")

    fut_wp_list = []
    curr_wp_dict = {
        "delta_time": 0.,
        "delta_frame": 0,
        "fut_anno_path": current_anno_path,
        "fut_rel_pos": [0.0, 0.0],
        "fut_world_pos": current_world_pos
    }
    fut_wp_list.append(curr_wp_dict)
    
    max_delta_frame = 0
    last_ft_wp = current_world_pos
    for key in wp_dict.keys():
        delta_time = float(key[:-1])
        delta_frame = int(delta_time * frame_rate)
        fut_anno_path = os.path.join(anno_path, f"{(frame_number + delta_frame):05d}.json.gz")
        fut_rel_pos = wp_dict[key]
        # print(f"[debug] delta_time = {delta_time}, delta_frame = {delta_frame}, position = {fut_rel_pos} fut_anno_path = {fut_anno_path}")
        fut_world_pos = transform_to_world_coordinates(fut_rel_pos + [0], current_inv_mat)
        # print(f"[debug] original world pos = {current_world_pos}, fut_world_pos = {fut_world_pos}")
        fut_wp_dict = {
            "delta_time": delta_time,
            "delta_frame": delta_frame,
            "fut_anno_path": fut_anno_path,
            "fut_rel_pos": fut_rel_pos,
            "fut_world_pos": fut_world_pos
        }
        if delta_frame > max_delta_frame:
            max_delta_frame = delta_frame
            last_ft_wp = fut_world_pos
        
        fut_wp_list.append(fut_wp_dict)

    wplen = len(fut_wp_list)

    ########################## Collisions ##########################

    all_collision_info = []
    collision_str = ""

    last_yaw = current_rotation[2]
    for i in range(1, wplen):
        distance_to_last = calculate_distance(fut_wp_list[i - 1]["fut_world_pos"], fut_wp_list[i]["fut_world_pos"])
        if distance_to_last > 0.1:
            yaw1 = calculate_yaw(fut_wp_list[i - 1]["fut_world_pos"], fut_wp_list[i]["fut_world_pos"])
            if i < wplen - 1:
                yaw2 = calculate_yaw(fut_wp_list[i]["fut_world_pos"], fut_wp_list[i + 1]["fut_world_pos"])
            else:
                yaw2 = yaw1
            fut_yaw = get_mean_angle(yaw1, yaw2)
            # print(f"[debug] yaw 1 = {yaw1}, yaw 2 = {yaw2}")
        else:
            fut_yaw = last_yaw
        # print(f"[debug] final_yaw = {fut_yaw}")
        collision_radius = 0.1
        collision_info = check_scene_overlap(ego_location=fut_wp_list[i]["fut_world_pos"],
                                             ego_yaw=fut_yaw, ego_extent=ego_extent,
                                             measurements=load_json_gz(fut_wp_list[i]["fut_anno_path"]),
                                             collision_radius=collision_radius, delta_time=delta_time)
        # print(f"[debug] collision_info = {collision_info}")
        all_collision_info.extend(collision_info)

        last_yaw = fut_yaw
    
    if len(all_collision_info) == 0:
        collision_str = "No collision happend. "
    else:
        collide_obj_dict = {}
        for event in all_collision_info:
            obj_id = event['obj_dict']['id']
            if obj_id not in collide_obj_dict:
                collide_obj_dict[obj_id] = event
        for obj_id, event in collide_obj_dict.items():
            collision_score *= collision_penalty
            collision_str += f"Collides with {event['obj_dict']['base_type']}(id={event['obj_dict']['id']}) " +\
                             f"{event['delta_time']}s later. "
    
    ########################## Efficiency ##########################
            
    last_anno_path = os.path.join(anno_path, f"{(frame_number + max_delta_frame):05d}.json.gz")
    last_bounding_boxes = load_json_gz(last_anno_path).get("bounding_boxes", [])
    last_ego_vehicle = next((item for item in last_bounding_boxes if item.get("class") == "ego_vehicle"), None)
    last_world_pos = last_ego_vehicle['location']
    gt_distance = calculate_distance(current_world_pos, last_world_pos)
    pred_distance = calculate_distance(current_world_pos, last_ft_wp)

    efficiency_str = "The efficiency is similar to ground truth. "
    efficiency_score = 1.0
    if gt_distance > 1.0:
        efficiency_score = min(1.0, float(pred_distance) / float(gt_distance))
        efficiency_str = f"The ego vehicle travels {round((efficiency_score * 100), 1)}% distance of gt vehicle. "
    
    final_score = collision_score * 0.6 + efficiency_score * 0.4
    final_reason = collision_str + efficiency_str
    final_score *= 100

    return final_score, final_reason
