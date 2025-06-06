import carla
import math
import numpy as np
import gzip
import os
import json
from shapely.geometry import Polygon, Point
import cv2
from scipy.spatial import KDTree
from .graph_utils import *
from webcolors import (
    CSS2_HEX_TO_NAMES,
    hex_to_rgb,
)
from .map_marks import *
from .hyper_params import *
from io_utils import print_debug
from itertools import islice

def convert_rgb_to_names(rgb_tuple):

    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS2_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    final_name = names[index]
    if final_name == 'black':
        r, g, b = rgb_tuple
        if 10 < r < 60 and g < 20 and b < 20 and (r - g) > 5 and (r - b) > 5:
            final_name = 'dark red'
        if r < 20 and 10 < g < 60 and b < 20 and (g - r) > 5 and (g - b) > 5:
            final_name = 'dark green'
        if r < 20 and g < 20 and 10 < b < 60 and (b - r) > 5 and (b - g) > 5:
            final_name = 'dark blue'
    return f'{final_name}'

# math utils

def calculate_distance(location1, location2):
    """
    Calculate Euclidean distance between two locations.
    """
    return math.sqrt((location1.x - location2.x) ** 2 +
                     (location1.y - location2.y) ** 2 +
                     (location1.z - location2.z) ** 2)

def calculate_2D_distance(location1, location2):
    """
    Calculate Euclidean distance between two locations, neglecting z-axis.
    """
    return math.sqrt((location1.x - location2.x) ** 2 +
                     (location1.y - location2.y) ** 2)

def compute_relative_velocity(ego_vehicle, actor):
    """
    Calculate actor's speed relative to the ego vehicle.

    Params:
    - actor: dict including "position (relative)", "speed", "yaw (relative, radian)".
    - ego_vehicle: dict including "speed".

    Return:
    - relative_speed: value of relative speed.
    - dot_product: negative for approaching, positive for moving away.
    """

    actor_yaw = actor["yaw"]
    actor_speed = actor["speed"]
    actor_velocity = np.array([actor_speed * np.cos(actor_yaw),
                               actor_speed * np.sin(actor_yaw)])


    ego_yaw = 0
    ego_speed = ego_vehicle["speed"]
    ego_velocity = np.array([ego_speed * np.cos(ego_yaw), 
                             ego_speed * np.sin(ego_yaw)])

    relative_velocity = actor_velocity - ego_velocity

    relative_position = np.array(actor["position"][:2])
    dot_product = np.dot(relative_velocity, relative_position)
    relative_speed = np.linalg.norm(relative_velocity)
   
    return float(relative_speed), float(dot_product)

def compute_intersection_distance(actor, ego_vehicle, deviate_degree=0, back_extent=1.5):
    """
    Computes the intersection point of the actor's velocity ray and the ego vehicle's velocity ray
    (both forward and backward). Returns the intersection point and its signed distance from the ego vehicle.

    Parameters:
    - actor: Dictionary containing:
        - "position": (x, y) position in the ego vehicle's coordinate system.
        - "speed": Scalar speed of the actor.
        - "yaw": Actor's orientation relative to the ego vehicle (in radians).
    - ego_vehicle: Dictionary containing:
        - "speed": Scalar speed of the ego vehicle.
    - deviate_degree: The yaw deviation angle (in degrees). Default set to 0.
    - back_extent: Distance to shift the actor's position backward along its yaw direction (default: 1.5)

    Returns:
    - intersection_point: (x, y) coordinates of the intersection in the ego vehicle's coordinate system.
    - signed_distance to ego: Signed distance from the ego vehicle to the intersection point.
      - Positive if the intersection is in the forward direction.
      - Negative if the intersection is in the backward direction.
      - None if the rays do not intersect.
    - distance to actor: Distance from the actor to the intersection point.
      - Positive if the intersection is in the forward direction.
      - None if the rays do not intersect.
    """

    actor_yaw = actor["yaw"]
    actor_speed = actor["speed"]
    actor_pos_center = np.array(actor["position"][:2])

    actor_dir_unit = np.array([np.cos(actor_yaw), np.sin(actor_yaw)])
    actor_pos = actor_pos_center - back_extent * actor_dir_unit # deviate it to the rear, which is safer
    actor_dir = actor_dir_unit * actor_speed

    deviate_radian = np.radians(deviate_degree)

    ego_yaw = deviate_radian
    ego_speed = ego_vehicle["speed"]
    ego_pos = np.array([0, 0])
    ego_dir = np.array([np.cos(ego_yaw), np.sin(ego_yaw)]) * ego_speed  # Velocity direction vector

    # Set up the equations: P_ego + t * D_ego = P_actor + s * D_actor
    A = np.column_stack((ego_dir, -actor_dir))
    b = actor_pos - ego_pos  # Target vector

    try:
        t, s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None, None  # The rays are parallel or do not intersect

    intersection_point = ego_pos + t * ego_dir

    return list(intersection_point), float(t * ego_speed), float(s * actor_speed)

def rotate_point(point, center, angle):
    """
    Rotate a 2D point around a center by a given angle (in radians).
    """
    x, y = point
    cx, cy = center
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    x -= cx
    y -= cy

    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta

    x_new += cx
    y_new += cy

    return [x_new, y_new]

def is_point_in_rotated_box(point, box_center, box_extent, box_yaw):
    """
    Check if a 2D point is inside a rotated rectangular box.
    """

    local_point = rotate_point(point, box_center, -box_yaw)

    dx, dy = box_extent
    if (
        -dx <= local_point[0] - box_center[0] <= dx and
        -dy <= local_point[1] - box_center[1] <= dy
    ):
        return True
    return False

def get_rotated_vertices(center, extent, yaw):
    """
    Calculate the 2D vertices of a rotated rectangle (collision box).
    """
    cx, cy = center
    hw, hh = extent[0] / 2, extent[1] / 2  # Half-width and half-height

    local_vertices = [
        [-hw, -hh],
        [-hw, hh],
        [hw, hh],
        [hw, -hh],
    ]

    rotated_vertices = []
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    for vx, vy in local_vertices:
        rx = cx + vx * cos_yaw - vy * sin_yaw
        ry = cy + vx * sin_yaw + vy * cos_yaw
        rotated_vertices.append([rx, ry])

    return rotated_vertices

def transform_to_ego_coordinates(world_coordinates, world2ego_matrix):
    """
    Transform a point in world coordinates to ego_vehicle coordinates.

    Params:
        world_coordinates (list or tuple): The (x, y, z) world coordinates.
        world2ego_matrix (list): The 4x4 world-to-ego transformation matrix.

    Returns:
        tuple: The transformed (x', y', z') coordinates in ego_vehicle's coordinate system.
    """

    P_world = np.array(list(world_coordinates) + [1])
    M_world2ego = np.array(world2ego_matrix)

    P_ego_homogeneous = M_world2ego @ P_world

    return tuple(P_ego_homogeneous[:3])

def transform_to_world_coordinates(ego_coordinates, world2ego_matrix):
    """
    Transform a point from ego_vehicle coordinates back to world coordinates.

    Params:
        ego_coordinates (list or tuple): The (x, y, z) coordinates in the ego_vehicle's coordinate system.
        world2ego_matrix (list): The 4x4 world-to-ego transformation matrix.

    Returns:
        tuple: The transformed (x', y', z') world coordinates.
    """

    P_ego = np.array(list(ego_coordinates) + [1])
    M_world2ego = np.array(world2ego_matrix)

    M_ego2world = np.linalg.inv(M_world2ego)

    P_world_homogeneous = M_ego2world @ P_ego

    return tuple(P_world_homogeneous[:3])

def rgb_to_color_name(rgb_str):
    """
    Convert an RGB string (e.g., '255,0,0') to a natural language color description.

    Params:
        rgb_str (str): RGB value in the format 'R,G,B'.

    Returns:
        str: The corresponding natural language color description.
    """

    try:
        r, g, b = map(int, rgb_str.split(","))
    except ValueError:
        return "unknown"

    return convert_rgb_to_names((r, g, b))

    # def euclidean_distance(rgb1, rgb2):
    #     return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)))

    # closest_color = min(color_mapping, key=lambda color: euclidean_distance((r, g, b), color_mapping[color]))

    # return closest_color

# calculation functions

def wps_next_until_lane_end(wp):
    try:
        road_id_cur = wp.road_id
        lane_id_cur = wp.lane_id
        road_id_next = road_id_cur
        lane_id_next = lane_id_cur
        curr_wp = [wp]
        next_wps = []
        # https://github.com/carla-simulator/carla/issues/2511#issuecomment-597230746
        while road_id_cur == road_id_next and lane_id_cur == lane_id_next:
            next_wp = curr_wp[0].next(1)
            if len(next_wp) == 0:
                break
            curr_wp = next_wp
            next_wps.append(next_wp[0])
            road_id_next = next_wp[0].road_id
            lane_id_next = next_wp[0].lane_id
    except:
        next_wps = []

    return next_wps

def print_waypoint_info(waypoint):
    """
    Print coordinate, road ID and lane ID of waypoint, just for debug
    """
    location = waypoint.transform.location
    road_id = waypoint.road_id
    lane_id = waypoint.lane_id

    print(f"Waypoint coordinates: x={location.x}, y={location.y}, z={location.z}")
    print(f"Route ID: {road_id}")
    print(f"Lane ID: {lane_id}")

def find_first_junction_in_direction(map, ego_location):
    """
    Finds the first junction in the direction of the ego vehicle.
    """
    waypoint = map.get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Driving)

    while waypoint and not waypoint.is_junction:
        # check whether it is junction or not every 2 meters
        waypoint = waypoint.next(2.0)[0] if waypoint.next(2.0) else None

    if waypoint and waypoint.is_junction:
        junction_location = waypoint.transform.location
        distance_to_junction = calculate_distance(ego_location, junction_location)
        return junction_location, distance_to_junction
    else:
        return None, None # represents no junction found

# def get_speed_limit(scene_data):
#     """
#     Finds the speed limit of ego
#     """
#     minimum_speed_limit = float('inf')

#     for actor in scene_data:
#         if 'traffic_sign' in actor['class']:
#             if actor['type_id'].startswith("traffic.speed_limit") and actor['affects_ego']:
#                 speed_limit = float(int(actor['type_id'].split('.')[-1]))
#                 if speed_limit < minimum_speed_limit:
#                     minimum_speed_limit = speed_limit

#     return minimum_speed_limit if minimum_speed_limit != float('inf') else 10000

def project_point_to_camera(intrinsic_matrix, world2cam_matrix, point, image_width, image_height):
    """
    Project point in world coordinate to camera 2d plane

    Returns:
        Tuple ([u, v], is_in_view) - Projection coordinate (u, v), bool indicates whether it is in camera's view
    """

    point_cam_location = (world2cam_matrix @ np.append(point, 1)).T[:3]

    pos_3d = np.array([point_cam_location[1], -point_cam_location[2], point_cam_location[0]])

    rvec = np.zeros((3, 1), np.float32)
    tvec = np.array([[0.0, 0.0, 0.0]], np.float32)
    dist_coeffs = np.zeros((5, 1), np.float32)

    points_2d, _ = cv2.projectPoints(
        pos_3d, rvec, tvec, intrinsic_matrix, dist_coeffs
    )
    u, v = points_2d[0][0]
    u, v = int(u), int(v)

    is_in_view = (pos_3d[2] > 0) and (0 <= u < image_width and 0 <= v < image_height)

    return [u, v], is_in_view

def get_vehicle_projected_corners(camera, vehicle):
    """
    Calculate vehicle's corners in camera's view

    Returns:
        - np.array which conveted from list of list [u, v] - Projection coordinate (u, v)
        - list of bool - Indicates whether it is in camera's view
    """
    intrinsic_matrix = np.array(camera['intrinsic'])
    world2cam_matrix = np.array(camera['world2cam'])
    image_width = camera['image_size_x']
    image_height = camera['image_size_y']

    if 'world_cord' in vehicle:
        vehicle_corners = np.array(vehicle['world_cord'])
    else:
        extent = vehicle.get('extent', [0.15, 0.15, 0.15])
        position = np.array(vehicle['center'])
        corners = np.array([
            [-extent[0], -extent[1], -extent[2]],  # bottom left back
            [extent[0], -extent[1], -extent[2]],   # bottom right back
            [extent[0], extent[1], -extent[2]],    # bottom right front
            [-extent[0], extent[1], -extent[2]],   # bottom left front
            [-extent[0], -extent[1], extent[2]],   # top left back
            [extent[0], -extent[1], extent[2]],    # top right back
            [extent[0], extent[1], extent[2]],     # top right front
            [-extent[0], extent[1], extent[2]]     # top left front
        ])

        world_yaw = vehicle.get('rotation', [0, 0, 0])[2]
        yaw_camera = world_yaw # - camera_yaw
        rotation_matrix = np.array([
            [np.cos(yaw_camera), -np.sin(yaw_camera), 0],
            [np.sin(yaw_camera), np.cos(yaw_camera), 0],
            [0, 0, 1]
        ])
        corners = corners @ rotation_matrix.T
        vehicle_corners = corners + position

    projected_corners = []
    in_view_list = []
    for corner in vehicle_corners:
        uv, is_in_view = project_point_to_camera(
            intrinsic_matrix, world2cam_matrix, corner, image_width, image_height
        )
        projected_corners.append(uv)
        in_view_list.append(is_in_view)

    return np.array(projected_corners), in_view_list

def is_vehicle_in_camera(camera, vehicle):
    """
    Check if the vehicle is in the view of camera
    """
    projected_corners, in_view_list = get_vehicle_projected_corners(camera, vehicle)

    for is_in_view in in_view_list:
        if is_in_view:
            return True

    # all vertices out of sight
    return False

def is_vehicle_in_junction(map, vehicle_location):
    """
    Check if the vehicle is at junction
    """
    waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

    if waypoint and waypoint.is_junction:
        return True
    else:
        return False

def get_lane_info(map, vehicle_location):
    """
    Get information keys of current lane
    """
    waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.libcarla.LaneType.Any)

    if not waypoint:
        return {}

    lane_direction = waypoint.lane_id
    leftmost_lane = lane_direction
    rightmost_lane = lane_direction
    lane_id_left_most_lane_same_direction = lane_direction
    vehicle_road_id = waypoint.road_id

    num_lanes_same_direction = 1
    num_lanes_opposite_direction = 0

    shoulder_left, parking_left, sidewalk_left, bikelane_left = False, False, False, False
    shoulder_right, parking_right, sidewalk_right, bikelane_right = False, False, False, False

    left_waypoint = waypoint.get_left_lane()
    while left_waypoint:
        if left_waypoint.lane_id == leftmost_lane or left_waypoint.road_id != vehicle_road_id:
            break

        lane_type = left_waypoint.lane_type
        if lane_type == carla.LaneType.Shoulder and left_waypoint.lane_width > 1.0:
            shoulder_left = True
        if lane_type == carla.LaneType.Parking:
            parking_left = True
        if lane_type == carla.LaneType.Sidewalk:
            sidewalk_left = True
        if lane_type == carla.LaneType.Biking:
            bikelane_left = True

        if left_waypoint.lane_id * lane_direction > 0:
            if lane_type == carla.LaneType.Driving:
                num_lanes_same_direction += 1
                lane_id_left_most_lane_same_direction = left_waypoint.lane_id
                # print_waypoint_info(left_waypoint)

            leftmost_lane = left_waypoint.lane_id
            left_waypoint = left_waypoint.get_left_lane()
        elif left_waypoint.lane_id * lane_direction < 0:
            if lane_type == carla.LaneType.Driving:
                num_lanes_opposite_direction += 1
                # print_waypoint_info(left_waypoint)

            leftmost_lane = left_waypoint.lane_id
            left_waypoint = left_waypoint.get_right_lane()

    right_waypoint = waypoint.get_right_lane()
    while right_waypoint:
        if right_waypoint.lane_id == rightmost_lane or right_waypoint.road_id != vehicle_road_id:
            break

        lane_type = right_waypoint.lane_type
        if lane_type == carla.LaneType.Shoulder and right_waypoint.lane_width > 1.0:
            shoulder_right = True
        if lane_type == carla.LaneType.Parking:
            parking_right = True
        if lane_type == carla.LaneType.Sidewalk:
            sidewalk_right = True
        if lane_type == carla.LaneType.Biking:
            bikelane_right = True

        if right_waypoint.lane_id * lane_direction > 0:
            if right_waypoint.lane_type == carla.LaneType.Driving:
                num_lanes_same_direction += 1
                # print_waypoint_info(right_waypoint)

            rightmost_lane = right_waypoint.lane_id
            right_waypoint = right_waypoint.get_right_lane()
        elif right_waypoint.lane_id * lane_direction < 0:
            if right_waypoint.lane_type == carla.LaneType.Driving:
                num_lanes_opposite_direction += 1
                # print_waypoint_info(right_waypoint)

            rightmost_lane = right_waypoint.lane_id
            right_waypoint = right_waypoint.get_left_lane()

    # get ego lane number counted from left to right
    # https://www.asam.net/standards/detail/opendrive/
    # most left should be always the smallest number
    ego_lane_number = abs(waypoint.lane_id - lane_id_left_most_lane_same_direction)

    ego_wp = waypoint
    # how far is next junction
    next_wps = wps_next_until_lane_end(ego_wp)
    try:
        next_lane_wps_ego = next_wps[-1].next(1)
        if len(next_lane_wps_ego) == 0:
            next_lane_wps_ego = [next_wps[-1]]
    except:
        next_lane_wps_ego = []
    if ego_wp.is_junction:
        distance_to_junction_ego = 0.0
        # get distance to ego vehicle
    elif len(next_lane_wps_ego)>0 and next_lane_wps_ego[0].is_junction:
        distance_to_junction_ego = next_lane_wps_ego[0].transform.location.distance(ego_wp.transform.location)
    else:
        distance_to_junction_ego = None

    next_road_ids_ego = []
    next_next_road_ids_ego = []
    for i, wp in enumerate(next_lane_wps_ego):
            next_road_ids_ego.append(wp.road_id)
            next_next_wps = wps_next_until_lane_end(wp)
            try:
                next_next_lane_wps_ego = next_next_wps[-1].next(1)
                if len(next_next_lane_wps_ego) == 0:
                    next_next_lane_wps_ego = [next_next_wps[-1]]
            except:
                next_next_lane_wps_ego = []
            for j, wp2 in enumerate(next_next_lane_wps_ego):
                if wp2.road_id not in next_next_road_ids_ego:
                    next_next_road_ids_ego.append(wp2.road_id)


    lane_info = {
        "num_lanes_same_direction": num_lanes_same_direction,
        "num_lanes_opposite_direction": num_lanes_opposite_direction,
        "ego_lane_number": ego_lane_number,
        'lane_change': waypoint.lane_change,
        'lane_change_str': str(waypoint.lane_change),
        'lane_type': waypoint.lane_type,
        'lane_type_str': str(waypoint.lane_type),
        'left_lane_marking_color': waypoint.left_lane_marking.color,
        'left_lane_marking_color_str': str(waypoint.left_lane_marking.color),
        'left_lane_marking_type': waypoint.left_lane_marking.type,
        'left_lane_marking_type_str': str(waypoint.left_lane_marking.type),
        'right_lane_marking_color': waypoint.right_lane_marking.color,
        'right_lane_marking_color_str': str(waypoint.right_lane_marking.color),
        'right_lane_marking_type': waypoint.right_lane_marking.type,
        'right_lane_marking_type_str': str(waypoint.right_lane_marking.type),
        "shoulder_left": shoulder_left,
        "parking_left": parking_left,
        "sidewalk_left": sidewalk_left,
        "bike_lane_left": bikelane_left,
        "shoulder_right": shoulder_right,
        "parking_right": parking_right,
        "sidewalk_right": sidewalk_right,
        "bike_lane_right": bikelane_right,
        'is_in_junction': ego_wp.is_junction,
        'junction_id': ego_wp.junction_id,
        'distance_to_junction': distance_to_junction_ego,
        'next_junction_id': next_lane_wps_ego[0].junction_id,
        'next_road_ids': next_road_ids_ego,
        'next_next_road_ids': next_next_road_ids_ego,
        'distance_to_junction_ego': distance_to_junction_ego,
        'next_junction_id_ego': next_lane_wps_ego[0].junction_id,
        'next_road_ids_ego': next_road_ids_ego,
        'next_next_road_ids_ego': next_next_road_ids_ego,
    }

    return lane_info

def get_road_and_lane_id(map, x, y, z):
    """
    Get the road_id and lane_id at a given (x, y, z) world coordinate in CARLA.
    
    Args:
        map (carla.Map): The CARLA map object.
        x (float): X coordinate in world space.
        y (float): Y coordinate in world space.
    
    Returns:
        tuple: (road_id, lane_id) if found, otherwise (None, None)
    """
    location = carla.Location(x=x, y=y, z=z)
    waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Any)
    
    if waypoint is not None:
        return waypoint.road_id, waypoint.lane_id
    else:
        return None, None

def get_relative_lane_id(map, x, y, z, n):
    """
    Get the road_id and lane_id for a lane that is n lanes away from the given (x, y, z) coordinate.

    Args:
        map (carla.Map): The CARLA map object.
        x (float): X coordinate in world space.
        y (float): Y coordinate in world space.
        z (float): Z coordinate in world space.
        n (int): Relative lane index (-2 means 2 lanes left, 1 means 1 lane right, 0 is current lane).

    Returns:
        tuple: (road_id, lane_id) of the closest matching lane.
    """
    location = carla.Location(x=x, y=y, z=z)
    waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Any)
    
    if waypoint is None:
        return None, None

    target_waypoint = waypoint

    direction = -1 if n < 0 else 1
    for _ in range(abs(n)):
        next_waypoint = target_waypoint.get_left_lane() if direction == -1 else target_waypoint.get_right_lane()
        if next_waypoint is not None:
            target_waypoint = next_waypoint
        else:
            break

    return target_waypoint.road_id, target_waypoint.lane_id

def get_nearest_point_on_lane(map, x, y, z, road_id, lane_id):
    """
    Find the closest point on a given road_id and lane_id to the input (x, y, z) coordinate.

    Args:
        map (carla.Map): The CARLA map object.
        x (float): X coordinate in world space.
        y (float): Y coordinate in world space.
        z (float): Z coordinate in world space.
        road_id (int): The target road ID.
        lane_id (int): The target lane ID.

    Returns:
        carla.Location: The closest point on the lane.
        Vector3D: The forward vector of this wp.
    """
    all_waypoints = map.generate_waypoints(1.0)

    valid_waypoints = [wp for wp in all_waypoints if wp.road_id == road_id and wp.lane_id == lane_id]

    if not valid_waypoints:
        return None, None

    closest_waypoint = min(valid_waypoints, key=lambda wp: calculate_2D_distance(carla.Location(x=x, y=y, z=z),
                                                                                 wp.transform.location))

    return closest_waypoint.transform.location, closest_waypoint.transform.get_forward_vector()

def get_z_rotation_from_forward_vector(forward_vector_x, forward_vector_y, forward_vector_z):
    """
    Compute the rotation (yaw) around the Z-axis from a given forward vector.

    Args:
        forward_vector (tuple): A 3D tuple (fx, fy, fz) representing the forward direction.

    Returns:
        float: Rotation in degrees around the Z-axis.
    """
    fx, fy, _ = forward_vector_x, forward_vector_y, forward_vector_z
    yaw_radians = math.atan2(fy, fx)
    yaw_degrees = math.degrees(yaw_radians)
    return yaw_degrees

def get_lane_deviate_info(map, vehicle):
    """
    Compute the deviation info of a vehicle.

    Args:
        vehicle (dict): vehicle info dictionary.
    Returns:
        road_yaw_deg (float): current waypoint's yaw in degrees.
        vehicle_yaw_deg (float): current waypoint's yaw in degrees.
        deviant_degree (float): vehicle's deviation from lane, + for clockwise (right).
    """
    nearest_road_forward_vec = get_nearest_waypoint_forward_vector(map, x=vehicle['location'][0],
                                                                        y=vehicle['location'][1],
                                                                        z=vehicle['location'][2])
    if nearest_road_forward_vec is None:
        return 0.0, 0.0, 0.0
    forward_yaw = get_z_rotation_from_forward_vector(forward_vector_x=nearest_road_forward_vec.x,
                                                        forward_vector_y=nearest_road_forward_vec.y,
                                                        forward_vector_z=nearest_road_forward_vec.z)
    
    ego_yaw_deg = forward_yaw
    road_yaw_deg = vehicle['rotation'][2]

    while road_yaw_deg < 0: road_yaw_deg += 360
    while road_yaw_deg >= 360: road_yaw_deg -= 360
    while ego_yaw_deg < 0: ego_yaw_deg += 360
    while ego_yaw_deg >= 360: ego_yaw_deg -= 360
    
    if road_yaw_deg <= ego_yaw_deg:
        left_deg = road_yaw_deg
        right_deg = road_yaw_deg + 360
    else:
        left_deg = road_yaw_deg - 360
        right_deg = road_yaw_deg
    
    turn_left = (ego_yaw_deg - left_deg) < (right_deg - ego_yaw_deg)

    if turn_left:
        return road_yaw_deg, ego_yaw_deg, ego_yaw_deg - left_deg
    else:
        return road_yaw_deg, ego_yaw_deg, ego_yaw_deg - right_deg

def world_point_deviate_info(target_x, target_y, vehicle):
    """
    Compute the deviation angle from ego vehicle to a world point.

    Args:
        target_x (float): target world x position.
        target_y (float): target world y position.
        vehicle (dict): dictionary with keys 'location' and 'rotation'.

    Returns:
        vehicle_yaw_deg (float): ego vehicle's yaw in degrees.
        target_yaw_deg (float): angle from vehicle to target in degrees.
        deviation_deg (float): deviation angle (target_yaw - vehicle_yaw), normalized to [-180, 180].
    """
    ego_x, ego_y = vehicle['location'][0], vehicle['location'][1]
    vehicle_yaw_deg = vehicle['rotation'][2]

    dx = target_x - ego_x
    dy = target_y - ego_y

    target_yaw_rad = math.atan2(dy, dx)
    target_yaw_deg = math.degrees(target_yaw_rad)

    vehicle_yaw_deg = vehicle_yaw_deg % 360
    target_yaw_deg = target_yaw_deg % 360

    deviation = target_yaw_deg - vehicle_yaw_deg
    if deviation > 180:
        deviation -= 360
    elif deviation < -180:
        deviation += 360

    return deviation

def find_last_non_junction_waypoint(map, x, y):
    """
    From given (x, y), trace backward along the road (every 0.5m) 
    until a junction is encountered. Return the last non-junction waypoint's (x, y).

    Args:
        map (carla.Map): CARLA map object.
        x (float): world x coordinate.
        y (float): world y coordinate.

    Returns:
        (float, float): (x, y) of the last non-junction waypoint.
    """
    waypoint = map.get_waypoint(carla.Location(x=x, y=y), project_to_road=True, lane_type=carla.LaneType.Driving)
    
    if waypoint is None:
        return None, None

    last_non_junction_wp = waypoint

    max_trial = 256
    num_trial = 0
    while num_trial < max_trial:
        prev_waypoints = waypoint.previous(0.5)
        if not prev_waypoints:
            break
        waypoint = prev_waypoints[0]
        if waypoint.is_junction:
            break
        last_non_junction_wp = waypoint
        num_trial += 1

    return last_non_junction_wp.transform.location.x, last_non_junction_wp.transform.location.y

def is_on_road(map, x, y, z):
    """
    Check if a given world coordinate (x, y, z) is on the road.

    Args:
        map (carla.Map): The CARLA map object.
        x (float): X coordinate in world space.
        y (float): Y coordinate in world space.
        z (float): Z coordinate in world space.

    Returns:
        bool: True if the location is on the road, False otherwise.
    """
    location = carla.Location(x, y, z)
    waypoint = map.get_waypoint(location, project_to_road=False)  # Do not project to nearest road
    return waypoint is not None

def get_nearest_waypoint_location(map, x, y, z):
    """
    Get the nearest waypoint's world coordinates.

    Args:
        map (carla.Map): The CARLA map object.
        x (float): X coordinate in world space.
        y (float): Y coordinate in world space.
        z (float): Z coordinate in world space.

    Returns:
        carla.Location or None: The nearest waypoint's location, or None if not found.
    """
    location = carla.Location(x, y, z)
    waypoint = map.get_waypoint(location, project_to_road=True)  # Project to the nearest road

    if waypoint:
        return waypoint.transform.location
    return None

def get_nearest_waypoint_forward_vector(map, x, y, z):
    """
    Get the nearest waypoint's world coordinates.

    Args:
        map (carla.Map): The CARLA map object.
        x (float): X coordinate in world space.
        y (float): Y coordinate in world space.
        z (float): Z coordinate in world space.

    Returns:
        Vector3D or None: The nearest waypoint's forward vector, or None if not found.
    """
    location = carla.Location(x, y, z)
    waypoint = map.get_waypoint(location, project_to_road=True)  # Project to the nearest road

    if waypoint:
        return waypoint.transform.get_forward_vector()
    return None

def get_other_vehicle_info(path, map, vehicle, ego):
    """
    Get information keys of other vehicle
    """

    vehicle_location = carla.Location(x=vehicle['location'][0], y=vehicle['location'][1], z=vehicle['location'][2])
    ego_location = carla.Location(x=ego['location'][0], y=ego['location'][1], z=ego['location'][2])

    vehicle_wp = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
    ego_wp = map.get_waypoint(ego_location, project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
    same_road_as_ego = False
    lane_relative_to_ego = None
    same_direction_as_ego = False

    next_wps = wps_next_until_lane_end(vehicle_wp)
    try:
        next_lane_wps = next_wps[-1].next(1)
        if len(next_lane_wps) == 0:
            next_lane_wps = [next_wps[-1]]
    except:
        next_lane_wps = []

    next_next_wps = []
    for i, wp in enumerate(next_lane_wps):
        next_next_wps = wps_next_until_lane_end(wp)

    try:
        next_next_lane_wps = next_next_wps[-1].next(1)
        if len(next_next_lane_wps) == 0:
            next_next_lane_wps = [next_next_wps[-1]]
    except:
        next_next_lane_wps = []

    if vehicle_wp.is_junction:
        distance_to_junction = 0.0
        # get distance to ego vehicle
    elif next_lane_wps and next_lane_wps[0].is_junction:
        distance_to_junction = next_lane_wps[0].transform.location.distance(vehicle_wp.transform.location)
    else:
        distance_to_junction = None

    next_road_ids = []
    for i, wp in enumerate(next_lane_wps):
        if wp.road_id not in next_road_ids:
            next_road_ids.append(wp.road_id)

    next_next_road_ids = []
    for i, wp in enumerate(next_next_lane_wps):
        if wp.road_id not in next_next_road_ids:
            next_next_road_ids.append(wp.road_id)

    left_wp, right_wp = ego_wp.get_left_lane(), ego_wp.get_right_lane()
    left_decreasing_lane_id = left_wp is not None and left_wp.lane_id < ego_wp.lane_id or right_wp is not None and right_wp.lane_id > ego_wp.lane_id

    remove_lanes_for_lane_relative_to_ego = 1
    wp = ego_wp
    is_opposite = False
    while True:
        flag = ego_wp.lane_id > 0 and left_decreasing_lane_id or ego_wp.lane_id < 0 and not left_decreasing_lane_id
        if is_opposite:
            flag = not flag
        wp = wp.get_left_lane() if flag else wp.get_right_lane()

        if wp is None or wp.lane_type == carla.LaneType.Driving and ego_wp.lane_id * wp.lane_id < 0:
            break

        is_opposite = ego_wp.lane_id * wp.lane_id < 0

        if wp.lane_type != carla.LaneType.Driving:
            remove_lanes_for_lane_relative_to_ego += 1

    if vehicle_wp.road_id == ego_wp.road_id:
        same_road_as_ego = True

        if vehicle_wp.lane_id * ego_wp.lane_id > 0:
            same_direction_as_ego = True

        lane_relative_to_ego = vehicle_wp.lane_id - ego_wp.lane_id
        lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1

        if not same_direction_as_ego:
            lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)

        lane_relative_to_ego = -lane_relative_to_ego

    try:
        rgb = tuple(map(int, vehicle.attributes['color'].split(',')))
        color_name = convert_rgb_to_names(rgb)
    except:
        rgb = None
        color_name = None

    lane_info = {
        'color_rgb': rgb,
        'color_name': color_name,
        'lane_type': vehicle_wp.lane_type,
        'lane_type_str': str(vehicle_wp.lane_type),
        'is_in_junction': vehicle_wp.is_junction,
        'junction_id': vehicle_wp.junction_id,
        'distance_to_junction': distance_to_junction,
        'next_junction_id': next_lane_wps[0].junction_id if next_lane_wps else -2,
        'next_road_ids': next_road_ids,
        'next_next_road_ids': next_next_road_ids,
        'same_road_as_ego': same_road_as_ego,
        'same_direction_as_ego': same_direction_as_ego,
        'lane_relative_to_ego': lane_relative_to_ego
    }

    return lane_info

def is_changing_lane(x, y, theta, map):
    """
    Judge whether the object is changing its lane by its position
    """

    vehicle_location = carla.Location(x=x, y=y)
    waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

    if not waypoint:
        return False

    if waypoint.is_junction:
        return False # vehicle must leave its lane at junction, so omit

    lane_yaw = waypoint.transform.rotation.yaw
    lane_direction = math.radians(lane_yaw)

    vehicle_direction = math.radians(theta)

    angle_diff = abs(vehicle_direction - lane_direction)
    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

    # if the car is more than CUT_IN_DEVIATION degrees off the lane
    angle_threshold = math.radians(CUT_IN_DEVIATION)
    if angle_diff > angle_threshold:
        return True

    # check the distance to neiboring lane, if too close
    left_lane = waypoint.get_left_lane()
    right_lane = waypoint.get_right_lane()

    if left_lane and left_lane.lane_type == carla.LaneType.Driving:
        distance_to_left_lane = left_lane.transform.location.distance(vehicle_location)
        if distance_to_left_lane * 0.5 <= waypoint.transform.location.distance(vehicle_location):
            return True

    if right_lane and right_lane.lane_type == carla.LaneType.Driving:
        distance_to_right_lane = right_lane.transform.location.distance(vehicle_location)
        if distance_to_right_lane * 0.5 <= waypoint.transform.location.distance(vehicle_location):
            return True

    return False

def is_ego_changing_lane(vehicle_data, map):
    """
    Judge whether the ego vehicle is changing its lane

    Params:
        vehicle_data (dict): from measurements
        carla_map (carla.Map): CARLA's map object
    """

    x, y = vehicle_data['x'], vehicle_data['y']
    theta = vehicle_data['theta']

    return is_changing_lane(x, y, theta, map)

def is_vehicle_changing_lane(vehicle_data, map):
    """
    Judge whether the vehicle is changing its lane

    Params:
        vehicle_data (dict): from scene_data (actor format)
        carla_map (carla.Map): CARLA's map object
    """

    x, y = vehicle_data['location'][0], vehicle_data['location'][1]
    theta = vehicle_data['rotation'][1] # yaw

    return is_changing_lane(x, y, theta, map)

def is_ego_changing_lane_due_to_obstacle(vehicle_data, map, bbox_list):
    """
    NOT USED!
    Judge whether the car is changing its lane due to obstacles

    Params:
        vehicle_data (dict): from measurements
        carla_map (carla.Map): CARLA map
        bbox_list (list): bbox measurement list
    """

    if is_ego_changing_lane(vehicle_data, map) is False:
        return False

    vehicle_location = carla.Location(x=vehicle_data['x'], y=vehicle_data['y'])
    vehicle_theta = vehicle_data['theta']
    waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    lane_direction = waypoint.transform.rotation.yaw

    angle_diff = (vehicle_theta - lane_direction) % 360
    if angle_diff > 180:
        angle_diff -= 360

    lane_change_direction = None
    angle_threshold = 10.0
    if angle_diff > angle_threshold:
        lane_change_direction = 'right'
    elif angle_diff < -angle_threshold:
        lane_change_direction = 'left'
    else:
        return False

    left_waypoint = waypoint.get_left_lane()
    right_waypoint = waypoint.get_right_lane()

    def calculate_distance_to_waypoint(wp):
        return math.sqrt((vehicle_location.x - wp.transform.location.x) ** 2 +
                         (vehicle_location.y - wp.transform.location.y) ** 2)

    nearest_lane = None
    nearest_lane_distance = INF_MAX # max value

    if left_waypoint:
        left_distance = calculate_distance_to_waypoint(left_waypoint)
        if left_distance < nearest_lane_distance:
            nearest_lane = left_waypoint
            nearest_lane_distance = left_distance

    if right_waypoint:
        right_distance = calculate_distance_to_waypoint(right_waypoint)
        if right_distance < nearest_lane_distance:
            nearest_lane = right_waypoint
            nearest_lane_distance = right_distance

    if nearest_lane is None:
        return False # there is no other lane!

    if lane_change_direction == 'left':
        if nearest_lane == left_waypoint:
            target_lane = left_waypoint
            original_lane = waypoint
        else:
            target_lane = waypoint
            original_lane = right_waypoint
    elif lane_change_direction == 'right':
        if nearest_lane == right_waypoint:
            target_lane = right_waypoint
            original_lane = waypoint
        else:
            target_lane = waypoint
            original_lane = left_waypoint

    detection_distance = 20.0  # detect obstacle
    angle_threshold = 10.0

    for obj in bbox_list:
        if obj['class'] == 'ego_vehicle':
            continue

        obj_location = obj['location']
        obj_x, obj_y, obj_z = obj_location[0], obj_location[1], obj_location[2]

        vehicle_x = vehicle_data['x']
        vehicle_y = vehicle_data['y']

        distance_to_obj = math.sqrt((obj_x - vehicle_x) ** 2 + (obj_y - vehicle_y) ** 2)

        if distance_to_obj > detection_distance:
            continue

        angle_to_obj = math.degrees(math.atan2(obj_y - vehicle_y, obj_x - vehicle_x))

        relative_angle = (angle_to_obj - vehicle_theta) % 360
        if relative_angle > 180:
            relative_angle -= 360

        if abs(relative_angle) < angle_threshold:
            obj_location = carla.Location(x=obj_x, y=obj_y, z=obj_z)
            obj_waypoint = map.get_waypoint(obj_location, project_to_road=True, lane_type=carla.LaneType.Driving)

            if obj_waypoint:
                if (obj_waypoint.road_id == original_lane.road_id and
                    obj_waypoint.lane_id == original_lane.lane_id):
                    return True

    return False

def is_vehicle_cutting_in(ego_data, vehicle_data, map, path):
    """
    Judge whether the car is cutting in ego's lane

    Params:
        ego_data (dict): from measurements
        vehicle_data (dict): from scene_data (actor)
        carla_map (carla.Map): CARLA map
    """

    vehicle_id = vehicle_data['id']
    ego_id = ego_data['id']

    old_lane_id, new_lane_id, vehicle_is_changing_lane, old_rel, new_rel = detect_lane_change_by_time(map, vehicle_id, ego_data, path)
    ego_old_lane_id, ego_new_lane_id, ego_is_changing_lane, _, _ = detect_lane_change_by_time(map, ego_id, ego_data, path)
    # print(f"[debug] lane_change, path = {path}, id = {vehicle_id}, original_lane = {old_lane_id}, new_lane = {new_lane_id}, old_rel = {old_rel}, new_rel = {new_rel}")
    # print(f"[debug] lane_change, path = {path}, ego_old_lane_id = {ego_old_lane_id}, ego_new_lane_id = {ego_new_lane_id}")

    if new_rel != 0 and vehicle_is_changing_lane is False:
        return False

    if (vehicle_data['position'][1] > -1.5 and vehicle_data['yaw'] < -0.05) or \
       (vehicle_data['position'][1] < 1.5 and vehicle_data['yaw'] > 0.05):

        ego_location = carla.Location(x=ego_data['location'][0], y=ego_data['location'][1])
        ego_waypoint = map.get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        return (vehicle_is_changing_lane and vehicle_data['road_id'] == ego_waypoint.road_id and new_lane_id == ego_waypoint.lane_id) or \
            (old_rel != 0 and new_rel == 0 and (ego_old_lane_id == ego_new_lane_id))

    else:
        return False

def detect_lane_change_by_time(map, id, ego, path, k=31):
    """
    Detect lane change by comparing lane_id and road_id across k frames.

    Args:
        path (str): Path to the current measurement file.
        k (int): Number of frames to consider before and after.

    Returns:
        tuple: (original_lane, new_lane, has_changed)
            original_lane: lane_id from k frames before (or earliest available).
            new_lane: lane_id from k frames after (or latest available).
            has_changed: bool, True if a lane change is detected, False otherwise.
    """
    dir_name, file_name = os.path.split(path)
    base_name, _ = os.path.splitext(file_name)
    base_name, _ = os.path.splitext(base_name)
    ext = ".json.gz"
    initial_index = int(base_name)

    def get_valid_measurement(index_offset):
        """Retrieve valid measurement at the given offset."""
        if index_offset < 0:
            offsets = range(index_offset, 0)
        else:
            offsets = range(index_offset, 0, -1)

        for offset in offsets:
            file_index = f"{initial_index + offset:05d}"
            file_path = os.path.join(dir_name, f"{file_index}{ext}")
            if os.path.exists(file_path):
                data = load_measurement(file_path)
                return data

    # Get earliest available (k frames before)
    original_data = get_valid_measurement(-k)
    if original_data is None:
        return None, None, False, None, None

    # Get latest available (k frames after)
    new_data = get_valid_measurement(k // 2)
    if new_data is None:
        return None, None, False, None, None

    now_data = load_measurement(path)
    if now_data is None:
        return None, None, False, None, None

    # Extract lane_id and road_id
    vehicle_past = find_dict_by_id(original_data, id)
    vehicle_future = find_dict_by_id(new_data, id)
    vehicle_now = find_dict_by_id(now_data, id)

    if vehicle_past is None or vehicle_future is None:
        return None, None, False, None, None

    original_lane = vehicle_past['lane_id']
    original_road = vehicle_past['road_id']
    original_dict = get_other_vehicle_info(path, map, vehicle_past, ego)
    new_lane = vehicle_future['lane_id']
    new_road = vehicle_future['road_id']
    new_dict = get_other_vehicle_info(path, map, vehicle_future, ego)

    # Check current waypoint's junction status
    now_location = carla.Location(x=vehicle_now['location'][0], y=vehicle_now['location'][1], z=vehicle_now['location'][2])
    if is_vehicle_in_junction(map, now_location):
        return None, None, False, None, None

    # Determine if lane change occurred
    has_changed = (
        original_lane != new_lane and
        original_road == new_road
    )

    return original_lane, new_lane, has_changed, original_dict['lane_relative_to_ego'], new_dict['lane_relative_to_ego']

def detect_future_lane_change(map, id, path, k=10):
    """
    Detect lane change by comparing lane_id and road_id across k frames.

    Args:
        path (str): Path to the current measurement file.
        k (int): Number of frames to consider before and after.

    Returns:
        tuple: (original_lane, new_lane, has_changed)
            original_lane: lane_id from k frames before (or earliest available).
            new_lane: lane_id from k frames after (or latest available).
            has_changed: bool, True if a lane change is detected, False otherwise.
    """

    # used by gt behaviour
    dir_name, file_name = os.path.split(path)
    base_name, _ = os.path.splitext(file_name)
    base_name, _ = os.path.splitext(base_name)
    ext = ".json.gz"
    initial_index = int(base_name)

    def get_valid_measurement(index_offset):
        """Retrieve valid measurement at the given offset."""
        if index_offset < 0:
            offsets = range(index_offset, 0)
        else:
            offsets = range(index_offset, 0, -1)

        for offset in offsets:
            file_index = f"{initial_index + offset:05d}"
            file_path = os.path.join(dir_name, f"{file_index}{ext}")
            if os.path.exists(file_path):
                data = load_measurement(file_path)
                return data

    # Get latest available (k frames after)
    new_data = get_valid_measurement(k)
    if new_data is None:
        return None, None, False

    now_data = load_measurement(path)
    if now_data is None:
        return None, None, False

    vehicle_future = find_dict_by_id(new_data, id)
    vehicle_now = find_dict_by_id(now_data, id)

    if vehicle_now is None or vehicle_future is None:
        return None, None, False

    original_lane = vehicle_now['lane_id']
    original_road = vehicle_now['road_id']
    new_lane = vehicle_future['lane_id']
    new_road = vehicle_future['road_id']

    # Check current waypoint's junction status
    now_location = carla.Location(x=vehicle_now['location'][0], y=vehicle_now['location'][1], z=vehicle_now['location'][2])
    if is_vehicle_in_junction(map, now_location):
        return None, None, False

    # Determine if lane change occurred
    has_changed = (
        original_lane != new_lane and
        original_road == new_road
    )

    return original_lane, new_lane, has_changed

def detect_future_lane_change_by_time(map, id, path, k=30):
    """
    Returns:

    original_lane (lane_id), new_lane (lane_id), has_changed (bool)
    """
    original_lane, new_lane, has_changed = None, None, False
    for i in range(1, k):
        original_lane, new_lane, has_changed = detect_future_lane_change(map, id, path, i)
        if has_changed:
            return original_lane, new_lane, has_changed
    return original_lane, new_lane, has_changed

def get_future_vehicle_dict_by_id(id, path, k=10):
    # used by gt behaviour
    # used by gt behaviour
    dir_name, file_name = os.path.split(path)
    base_name, _ = os.path.splitext(file_name)
    base_name, _ = os.path.splitext(base_name)
    ext = ".json.gz"
    initial_index = int(base_name)

    def get_valid_measurement(index_offset):
        """Retrieve valid measurement at the given offset."""
        if index_offset < 0:
            offsets = range(index_offset, 0)
        else:
            offsets = range(index_offset, 0, -1)

        for offset in offsets:
            file_index = f"{initial_index + offset:05d}"
            file_path = os.path.join(dir_name, f"{file_index}{ext}")
            if os.path.exists(file_path):
                data = load_measurement(file_path)
                return data

    new_data = get_valid_measurement(k)
    if new_data is None:
        return None

    vehicle_future = find_dict_by_id(new_data, id)

    return vehicle_future

def get_future_measurements(path, k=10):
    # used by gt behaviour
    dir_name, file_name = os.path.split(path)
    base_name, _ = os.path.splitext(file_name)
    base_name, _ = os.path.splitext(base_name)
    ext = ".json.gz"
    initial_index = int(base_name)

    def get_valid_measurement(index_offset):
        """Retrieve valid measurement at the given offset."""
        if index_offset < 0:
            offsets = range(index_offset, 0)
        else:
            offsets = range(index_offset, 0, -1)

        for offset in offsets:
            file_index = f"{initial_index + offset:05d}"
            file_path = os.path.join(dir_name, f"{file_index}{ext}")
            if os.path.exists(file_path):
                # print(f"retrieved measurement from {file_path}")
                data = load_measurement(file_path)
                return data
        return None

    new_data = get_valid_measurement(k)

    return new_data

def get_ego_vehicle_location_from_measurements(measurements):
    bounding_boxes = measurements.get("bounding_boxes", [])
    ego_vehicle = next((item for item in bounding_boxes if item.get("class") == "ego_vehicle"), None)
    location = ego_vehicle['location']
    
    return location

def find_walker_location(path):
    """
    Find the location of a walker that meets the specified conditions.

    Parameters:
        path (str): Path to the initial file.

    Returns:
        list or None: The location of the walker that meets the conditions, or None if not found.
    """
    dir_name, file_name = os.path.split(path)
    base_name, _ = os.path.splitext(file_name)
    base_name, _ = os.path.splitext(base_name)
    ext = ".json.gz"
    initial_index = int(base_name)

    current_index = initial_index

    while True:
        file_index = f"{current_index:05d}"
        file_path = os.path.join(dir_name, f"{file_index}{ext}")

        if not os.path.exists(file_path):
            return None  # File does not exist, terminate the search

        data = load_measurement(file_path)
        bounding_boxes = data.get("bounding_boxes", [])

        # Find ego_vehicle's location
        ego_vehicle = next((item for item in bounding_boxes if item.get("class") == "ego_vehicle"), None)
        if ego_vehicle is None:
            current_index += 1
            continue  # Skip to the next file if ego_vehicle is not found

        ego_z = ego_vehicle["location"][2]

        # Check for a walker within the distance condition
        for item in bounding_boxes:
            if item.get("class") == "walker" and abs(item["location"][2] - ego_z) <= 20:
                return item["location"]

        current_index += 1

def get_acceleration_by_imu(speed, theta, acceleration):
    """
    Returns:
        str: "Accelerating" or
             "Decelerating" or
             "Constant Speed"
    """
    if len(acceleration) < 2:
        raise ValueError("Acceleration vector must have at least two components [ax, ay].")

    ax, ay = acceleration[0], acceleration[1]

    theta_rad = math.radians(theta)
    acc_in_direction = ax * math.cos(theta_rad) + ay * math.sin(theta_rad)

    epsilon = 1e-2

    if abs(acc_in_direction) < epsilon:
        return "Constant Speed"
    elif acc_in_direction * speed > 0:
        return "Accelerating"
    else:
        return "Decelerating"

def load_measurement(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def find_location_by_id(data, target_id):
    for obj in data.get('bounding_boxes', []):
        if obj.get('id') == target_id:
            return obj.get('location'), obj.get('rotation')
    return None, None

def find_dict_by_id(data, target_id):
    for obj in data.get('bounding_boxes', []):
        if obj.get('id') == target_id:
            return obj
    return None

def get_acceleration_by_future(path, k):
    """
    Calculate acceleration trend by future k measurements

    Returns:
        str: "Accelerate", "Decelerate", "Constant", "Ambiguous"
    """

    dir_name, file_name = os.path.split(path)
    base_name, _ = os.path.splitext(file_name)
    base_name, _ = os.path.splitext(base_name)
    ext = ".json.gz"
    initial_index = int(base_name)

    speeds = []
    for i in range(k + 1):
        file_index = f"{initial_index + i:05d}"
        file_path = os.path.join(dir_name, f"{file_index}{ext}")

        if not os.path.exists(file_path):
            break

        data = load_measurement(file_path)
        speeds.append(data['speed'])

    if len(speeds) < 2:
        return "Ambiguous"

    acceleration_trend = speeds[-1] - speeds[0]
    if acceleration_trend > 0:
        return "Accelerate"
    elif acceleration_trend < 0:
        return "Decelerate"
    else:
        return "Constant"

def get_steer_by_future(path, id, k=5):
    """
    Calculate steer trend by future k measurements based on angle and distance.

    Args:
        path (str): Path to the current measurement file.
        k (int): Number of future measurements to consider.
        id (int): id attribute of vehicle.

    Returns:
        float: Estimated steer curvature (negative for left, positive for right, zero for straight).
    """
    # we use 1 / curvative radius as steer
    # negative means left, positive means right

    dir_name, file_name = os.path.split(path)
    base_name, _ = os.path.splitext(file_name)
    base_name, _ = os.path.splitext(base_name)
    ext = ".json.gz"
    initial_index = int(base_name)

    # Load current and future measurement
    current_file = os.path.join(dir_name, f"{initial_index:05d}{ext}")
    future_file = os.path.join(dir_name, f"{initial_index + k:05d}{ext}")

    if not (os.path.exists(current_file) and os.path.exists(future_file)):
        return 0.0  # If not enough data, return 0.0

    # Get locations and rotations
    current_data = load_measurement(current_file)
    future_data = load_measurement(future_file)

    current_location, current_rotation = find_location_by_id(current_data, id)
    future_location, future_rotation = find_location_by_id(future_data, id)

    if current_location is None or future_location is None:
        return 0.0  # If locations are missing, return 0.0

    # Convert locations to numpy arrays
    current_location = np.array(current_location)
    future_location = np.array(future_location)

    # Calculate distance and angle change
    distance = np.linalg.norm(future_location - current_location)
    current_angle = np.radians(current_rotation[2])  # Convert yaw to radians
    future_angle = np.radians(future_rotation[2])  # Convert yaw to radians
    angle_change = future_angle - current_angle

    # Normalize angle to range [-pi, pi]
    angle_change = np.arctan2(np.sin(angle_change), np.cos(angle_change))

    if distance < 1e-3 or (-1e-6 < angle_change < 1e-6):
         return 0.0  # Straight path

    # Calculate curvature radius and its inverse
    curvature_radius = distance / abs(angle_change)
    curvature = 10.0 / curvature_radius

    # Determine sign based on angle change
    steer = curvature if angle_change > 0 else -curvature

    return steer

def get_affect_flags(bbox_data):
    """
    Analyze traffic and get affect flags

    Params:
        bbox_data (list): bounding_box data from measurements

    Returns:
        dict: flags
    """

    flags = {
        "affected_by_red_light": False,
        "affected_by_yellow_light": False,
        "affected_by_stop_sign": False,
        "traffic_light_state": None
    }

    # for traffic_light
    # 0 - Red; 1 - Yellow; 2 - Green; 3 - Off; 4 - Unknown;
    for actor in bbox_data:
        if "traffic_light" in actor["class"]:
            if actor["state"] == 2 and actor["affects_ego"] is True:
                flags["traffic_light_state"] = "green"
            if actor["state"] == 1 and actor["affects_ego"] is True:
                flags["affected_by_yellow_light"] = True
                flags["traffic_light_state"] = "yellow"
            if actor["state"] == 0 and actor["affects_ego"] is True:
                flags["affected_by_red_light"] = True
                flags["traffic_light_state"] = "red"

        if "traffic_sign" in actor["class"]:
            if "stop" in actor["type_id"].lower() and actor["affects_ego"] is True:
                flags["affected_by_stop_sign"] = True

    return flags

def get_walker_hazard_with_prediction(bbox_data, expansion=1.2, prediction_time=20.0, delta_time=0.1):
    """
    Determine if any walker will enter the ego_vehicle's collision box within the prediction time
    and return a list of IDs of walkers with collision risks.

    Params:
        bbox_data (list): List of dictionaries containing data for all actors.
        expansion (float): Scaling factor for the collision box, default is 1.2 (expand by 20%).
        prediction_time (float): Time window (in seconds) for collision prediction, default is 20.
        delta_time (float): Time step for prediction intervals, default is 0.1 seconds.

    Returns:
        list: List of walker IDs with collision risks.
    """

    hazardous_walkers = []

    ego_vehicle = next((actor for actor in bbox_data if actor["class"] == "ego_vehicle"), None)
    if not ego_vehicle:
        return hazardous_walkers

    ego_center = ego_vehicle["location"][:2]
    ego_extent = ego_vehicle["extent"]
    ego_speed = ego_vehicle.get("speed", 0)  # Default speed is 0
    ego_yaw = math.radians(ego_vehicle["rotation"][2])  # Extract yaw angle and convert to radians

    ego_expanded_extent = [ego_extent[0] * expansion, ego_extent[1] * expansion]

    ego_future_position = [
        ego_center[0] + ego_speed * prediction_time * math.cos(ego_yaw),
        ego_center[1] + ego_speed * prediction_time * math.sin(ego_yaw),
    ]

    for actor in bbox_data:
        if actor["class"] == "walker":
            walker_id = actor["id"]
            walker_position = actor["location"][:2]  # Extract x, y coordinates
            walker_speed = actor.get("speed", 0)
            walker_yaw = math.radians(actor["rotation"][2])  # yaw

            # Simulate motion over time and check for collision
            time_elapsed = 0
            collision_detected = False
            while time_elapsed <= prediction_time:
                ego_future_position = [
                    ego_center[0] + ego_speed * time_elapsed * math.cos(ego_yaw),
                    ego_center[1] + ego_speed * time_elapsed * math.sin(ego_yaw),
                ]
                walker_future_position = [
                    walker_position[0] + walker_speed * time_elapsed * math.cos(walker_yaw),
                    walker_position[1] + walker_speed * time_elapsed * math.sin(walker_yaw),
                ]

                if is_point_in_rotated_box(walker_future_position, ego_future_position, ego_expanded_extent, ego_yaw):
                    collision_detected = True
                    break

                time_elapsed += delta_time

            if collision_detected:
                hazardous_walkers.append(walker_id)

    return hazardous_walkers

def get_all_hazard_with_prediction_sorted(bbox_data, expansion=1.2, prediction_time=20.0, delta_time=0.1):
    """
    Predict if the ego_vehicle will collide with any actor within the prediction time,
    using detailed collision box checks for all vertices, and sort the result by distance.

    Params:
        bbox_data (list): List of dictionaries containing data for all actors.
        expansion (float): Scaling factor for the ego_vehicle's collision box, default is 1.2 (expand by 20%).
        prediction_time (float): Time window (in seconds) for collision prediction, default is 20.
        delta_time (float): Time step for prediction intervals, default is 0.1 seconds.

    Returns:
        list: List of dictionaries with 'id' and 'distance' keys for hazardous actors, sorted by distance.
    """

    # Find the ego_vehicle data
    ego_vehicle = next((actor for actor in bbox_data if actor["class"] == "ego_vehicle"), None)
    if not ego_vehicle:
        return []

    # Extract ego_vehicle information
    ego_center = ego_vehicle["location"][:2]
    ego_extent = [ego_vehicle["extent"][0] * expansion, ego_vehicle["extent"][1] * expansion]
    ego_speed = ego_vehicle.get("speed", 0)
    ego_yaw = math.radians(ego_vehicle["rotation"][2])

    hazardous_actors = []

    # Iterate through all other actors
    for actor in bbox_data:
        if actor["class"] == "ego_vehicle":
            continue

        actor_center = actor["location"][:2]
        actor_extent = [actor["extent"][0], actor["extent"][1]]
        actor_speed = actor.get("speed", 0)
        actor_yaw = math.radians(actor["rotation"][2])

        time_elapsed = 0
        collision_detected = False
        while time_elapsed <= prediction_time:
            ego_future_position = [
                ego_center[0] + ego_speed * time_elapsed * math.cos(ego_yaw),
                ego_center[1] + ego_speed * time_elapsed * math.sin(ego_yaw),
            ]
            actor_future_position = [
                actor_center[0] + actor_speed * time_elapsed * math.cos(actor_yaw),
                actor_center[1] + actor_speed * time_elapsed * math.sin(actor_yaw),
            ]

            ego_vertices = get_rotated_vertices(ego_future_position, ego_extent, ego_yaw)
            actor_vertices = get_rotated_vertices(actor_future_position, actor_extent, actor_yaw)

            if any(is_point_in_rotated_box(vertex, actor_future_position, actor_extent, actor_yaw) for vertex in ego_vertices) or \
               any(is_point_in_rotated_box(vertex, ego_future_position, ego_extent, ego_yaw) for vertex in actor_vertices):
                collision_detected = True
                break

            time_elapsed += delta_time

        if collision_detected:
            hazardous_actors.append(actor)

    # Sort the actors by distance
    hazardous_actors.sort(key=lambda x: x["distance"])

    return hazardous_actors

def vehicle_obstacle_list(ego_data, vehicle_list, carla_map, max_distance, junction_threshold):
    """
    Helper function to detect all vehicles in front of the agent blocking its path, sorted by distance.

    Args:
        ego_data (dict): Ego vehicle object information.
        vehicle_list (list): List containing vehicle object information.
        carla_map (carla.map): Current map.
        max_distance (float): Max freespace to check for obstacles.
        junction_threshold (float): When vehicle waypoint goes into junction, how much further is detected.

    Returns:
        Tuple (bool, list): A bool indicating if any vehicle is detected and a list of all blocking vehicles sorted by distance.
    """

    def compute_distance(loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    def get_forward_vector(rotation):
        theta = math.radians(rotation[1])
        return [math.cos(theta), math.sin(theta)]

    ego_location = ego_data['location']
    ego_rotation = ego_data['rotation']
    ego_extent = ego_data['extent']
    ego_forward_vector = get_forward_vector(ego_rotation)

    ego_wpt = carla_map.get_waypoint(
        carla.Location(x=ego_location[0], y=ego_location[1], z=ego_location[2]),
        lane_type=carla.LaneType.Any
    )

    search_distance = max_distance

    def get_route_polygon():
        extent_y = ego_extent[1]
        r_ext = extent_y
        l_ext = -extent_y

        right_vector = [-ego_forward_vector[1], ego_forward_vector[0]]

        p1 = (ego_location[0] + r_ext * right_vector[0], ego_location[1] + r_ext * right_vector[1])
        p2 = (ego_location[0] + l_ext * right_vector[0], ego_location[1] + l_ext * right_vector[1])

        forward_points = []
        total_distance = 0
        current_wpt = ego_wpt
        last_wpt = None
        if ego_wpt.is_junction:
            last_wpt = ego_wpt
        else:
            while total_distance <= search_distance:
                # print(f"[debug] current_wpt = {current_wpt}, current_wpt.next(2.0) = {current_wpt.next(2.0)}")
                next_wpt_list = current_wpt.next(2.0)
                if not next_wpt_list or len(next_wpt_list) == 0:
                    break
                next_wpt = next_wpt_list[0]
                total_distance += 2.0
                if not next_wpt:
                    break

                if next_wpt.is_junction:
                    last_wpt = next_wpt
                    break

                forward_location = next_wpt.transform.location
                forward_vector = next_wpt.transform.get_right_vector()
                forward_p1 = (
                    forward_location.x + r_ext * forward_vector.x,
                    forward_location.y + r_ext * forward_vector.y
                )
                forward_p2 = (
                    forward_location.x + l_ext * forward_vector.x,
                    forward_location.y + l_ext * forward_vector.y
                )

                forward_points.append(forward_p1)
                forward_points.append(forward_p2)
                current_wpt = next_wpt

        route_bb = [p1, p2] + forward_points

        if len(route_bb) < 3:  # can't form a polygon
            return None, None
        return Polygon(route_bb), last_wpt

    route_polygon, last_wpt = get_route_polygon()

    detected_vehicles = []

    for target_vehicle in vehicle_list:
        if target_vehicle['id'] == ego_data['id']:
            continue

        if target_vehicle['lane_id'] != ego_data['lane_id']:
            # not in original code, but when route curves,
            # sometimes polygon produces error
            continue

        target_location = target_vehicle['location']

        if 'world_cord' in target_vehicle and target_vehicle['world_cord']:
            world_corners = np.array(target_vehicle['world_cord'])
        else:
            extent = target_vehicle.get('extent', [0.15, 0.15, 0.15])
            position = np.array(target_vehicle['center'])
            corners = np.array([
                [-extent[0], -extent[1], -extent[2]],  # bottom left back
                [extent[0], -extent[1], -extent[2]],   # bottom right back
                [extent[0], extent[1], -extent[2]],    # bottom right front
                [-extent[0], extent[1], -extent[2]],   # bottom left front
                [-extent[0], -extent[1], extent[2]],   # top left back
                [extent[0], -extent[1], extent[2]],    # top right back
                [extent[0], extent[1], extent[2]],     # top right front
                [-extent[0], extent[1], extent[2]]     # top left front
            ])
            world_yaw = target_vehicle.get('rotation', [0, 0, 0])[2]

            yaw_camera = world_yaw

            rotation_matrix = np.array([
                [np.cos(yaw_camera), -np.sin(yaw_camera), 0],
                [np.sin(yaw_camera), np.cos(yaw_camera), 0],
                [0, 0, 1]
            ])
            corners = corners @ rotation_matrix.T
            world_corners = corners + position

        target_bounding_box = world_corners.tolist()

        if compute_distance(ego_location, target_location) > search_distance:
            continue

        target_polygon = Polygon([(v[0], v[1]) for v in target_bounding_box])

        if route_polygon and route_polygon.intersects(target_polygon):
            detected_vehicles.append(target_vehicle)
            continue

        if last_wpt:
            target_forward_vector = get_forward_vector(target_vehicle['rotation'])
            last_forward_vector = last_wpt.transform.get_forward_vector()
            dot_product = last_forward_vector.x * target_forward_vector[0] + last_forward_vector.y * target_forward_vector[1]
            dot_product = max(-1, min(dot_product, 1))
            angle = math.degrees(math.acos(dot_product))
            distance = compute_distance([last_wpt.transform.location.x, last_wpt.transform.location.y], target_location)

            if distance < junction_threshold and abs(angle) <= 20:
                detected_vehicles.append(target_vehicle)

    detected_vehicles.sort(key=lambda v: v['distance'])

    return bool(detected_vehicles) and len(detected_vehicles) > 0, detected_vehicles

def vehicle_obstacle_detected(ego_data, vehicle_list, carla_map, max_distance=30.0, junction_threshold=3.0):
    """
    Method to check if there is a vehicle in front of the agent blocking its path.
    Migrated from pdm_lite, but massive modification
        :param ego_vehicle (actor dict): ego vehicle object information.
        :param vehicle_list (list of actor dict): list contatining vehicle object information.
        :param carla_map (carla.map): current map.
        :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        :param junction_threshold: when vehicle waypoint goes into junction, how much further is detected.
                If None, the base threshold value is used
        :return Tuple (bool: detected or not, dict(actor): nearest obstacle vehicle at front, if exist)
    """

    detected, vehicle_list = vehicle_obstacle_list(ego_data, vehicle_list, carla_map, max_distance, junction_threshold)
    return (detected, vehicle_list[0]) if detected else (False, [])

def remove_normal_speed_vehicles(vehicle_list, ego_vehicle):
    """
    Remove strings containing my_str from the list and filter out elements that do not meet the speed condition.

    Args:  
        vehicle_list (list): A list of vehicles, where each element is a dictionary.  
        my_str (str): A substring used for string matching.  
        ego_vehicle (dict): Contains the speed and rotation information of the ego vehicle.  

    Returns:  
        list: The filtered list of vehicles.  
    """
    def speed_in_direction(vehicle, ego_vehicle):
        def rotation_to_vector(rotation):
            yaw = np.deg2rad(rotation[2])
            return np.array([np.cos(yaw), np.sin(yaw)])

        ego_dir = rotation_to_vector(ego_vehicle["rotation"])
        hazard_dir = rotation_to_vector(vehicle["rotation"])

        if not vehicle.get("speed"):
            return 0.0
        hazard_speed_in_ego_dir = np.dot(hazard_dir, ego_dir) * vehicle["speed"]
        return hazard_speed_in_ego_dir

    return [
        vehicle for vehicle in vehicle_list
        if speed_in_direction(vehicle, ego_vehicle) < max(0.5, ego_vehicle["speed"] * (1 / 2))
    ]

def get_hazard_by_future(path, map, k, filter=None, max_distance=30.0, junction_threshold=3.0):
    """
    Identify hazards within the next k measurements based on the specified filter.

    Args:
        path (str): Path to the current measurement file.
        map (carla.Map): Current map.
        k (int): Number of future measurements to consider.
        filter (str or None): Substring to filter vehicle class. If None, consider all vehicles.
        max_distance (float): Maximum detection distance for obstacles.
        junction_threshold (float): Threshold for detecting vehicles at junctions.

    Returns:
        list: Sorted list of actor dicts representing hazards, ordered by distance.
    """
    dir_name, file_name = os.path.split(path)
    base_name, _ = os.path.splitext(file_name)
    base_name, _ = os.path.splitext(base_name)
    ext = ".json.gz"
    initial_index = int(base_name)

    hazards = []

    current_file = os.path.join(dir_name, f"{initial_index:05d}{ext}")
    if not os.path.exists(current_file):
        return hazards

    current_data = load_measurement(current_file)
    ego_data = next((item for item in current_data["bounding_boxes"] if item["class"] == "ego_vehicle"), None)
    if ego_data is None:
        return hazards

    carla_map = map

    # Iterate over future frames
    for i in range(k + 1):
        future_file = os.path.join(dir_name, f"{initial_index + i:05d}{ext}")
        if not os.path.exists(future_file):
            break

        future_data = load_measurement(future_file)
        vehicle_list = [
            item for item in future_data["bounding_boxes"]
            if filter is None or filter in item["class"]
        ]

        future_ego_data = next((item for item in future_data["bounding_boxes"] if item["class"] == "ego_vehicle"), None)
        if future_ego_data is None:
            break

        # Call vehicle_obstacle_list to get hazards
        _, hazard_list = vehicle_obstacle_list(
            ego_data=ego_data,
            vehicle_list=vehicle_list,
            carla_map=carla_map,
            max_distance=max_distance * (i + 1) / (k),
            junction_threshold=junction_threshold
        )

        append_list = []
        for actor in hazard_list:
            actor['position'] = transform_to_ego_coordinates(actor['location'], future_ego_data['world2ego'])
            if actor['position'][0] > 0.0: # in the front
                append_list.append(actor)
        hazards.extend(append_list)

    unique_ids = {hazard["id"] for hazard in hazards}

    matched_hazards = []
    for bbox in current_data["bounding_boxes"]:
        if bbox["id"] in unique_ids:
            matched_hazards.append(bbox)

    matched_hazards.sort(key=lambda x: x["distance"])

    return remove_normal_speed_vehicles(matched_hazards, ego_data)

def get_hazard_by_current_frame(current_data, map, filter=None, max_distance=30.0, junction_threshold=3.0):
    """
    Identify hazards at current frame within max_distance based on the specified filter.

    Args:
        current_data (dict): The current measurement dictionary.
        map (carla.Map): Current map.
        filter (str or None): Substring to filter vehicle class. If None, consider all vehicles.
        max_distance (float): Maximum detection distance for obstacles.
        junction_threshold (float): Threshold for detecting vehicles at junctions.

    Returns:
        list: Sorted list of actor dicts representing hazards, ordered by distance.
    """
    
    hazards = []
    ego_data = next((item for item in current_data["bounding_boxes"] if item["class"] == "ego_vehicle"), None)
    if ego_data is None:
        return hazards
    
    vehicle_list = [
        item for item in current_data["bounding_boxes"]
        if filter is None or filter in item["class"]
    ]

    carla_map = map

    _, hazard_list = vehicle_obstacle_list(
        ego_data=ego_data,
        vehicle_list=vehicle_list,
        carla_map=carla_map,
        max_distance=max_distance,
        junction_threshold=junction_threshold
    )

    append_list = []
    for actor in hazard_list:
        actor['position'] = transform_to_ego_coordinates(actor['location'], ego_data['world2ego'])
        if actor['position'][0] > 0.0: # in the front
            append_list.append(actor)
    hazards.extend(append_list)

    unique_ids = {hazard["id"] for hazard in hazards}

    matched_hazards = []
    for bbox in current_data["bounding_boxes"]:
        if bbox["id"] in unique_ids:
            matched_hazards.append(bbox)

    matched_hazards.sort(key=lambda x: x["distance"])

    return remove_normal_speed_vehicles(matched_hazards, ego_data)

def get_pos_str(actor_rel_position):
    x, y = actor_rel_position[0], actor_rel_position[1]

    angle = math.degrees(math.atan2(y, x))

    if angle < 0:
        angle += 360

    # don't modify 'to'! because these are both used in direction desc and overlap drive directions
    if (345 <= angle or angle < 15) and abs(y) < 1.5:
        rough_pos_str = 'to the front'
    elif 75 <= angle < 105 and abs(x) < 1.5:
        rough_pos_str = 'to the right'
    elif 165 <= angle < 195 and abs(y) < 1.5:
        rough_pos_str = 'to the rear'
    elif 255 <= angle < 285 and abs(x) < 1.5:
        rough_pos_str = 'to the left'
    elif 0 <= angle < 90:
        rough_pos_str = 'to the front right'
    elif 90 <= angle < 180:
        rough_pos_str = 'to the rear right'
    elif 180 <= angle < 270:
        rough_pos_str = 'to the rear left'
    elif 270 <= angle < 360:
        rough_pos_str = 'to the front left'
    else:
        rough_pos_str = 'at an unknown position'

    return rough_pos_str

def vehicle_is_too_dangerous(other_vehicle):
    if 'num_points' not in other_vehicle:
        return False
    if other_vehicle['num_points'] <= 0:
        return False
    if 'distance' not in other_vehicle:
        return False
    return other_vehicle['position'][0] > TOO_DANGEROUS_X_MIN and \
        -TOO_DANGEROUS_Y_MAX < other_vehicle['position'][1] < TOO_DANGEROUS_Y_MAX and \
        other_vehicle['distance'] < TOO_DANGEROUS_DISTANCE_MAX

def get_default_overlap_reason(other_vehicle):
    dir_str = get_pos_str(other_vehicle['position'])
    if vehicle_is_too_dangerous(other_vehicle):
        f"does not stop immediately"
    return f"drives too fast without avoiding the potential danger {dir_str} when getting closer"

def is_vehicle_pointing_towards_ego(vehicle_position, vehicle_yaw, threshold=45.0):
    # Get the position of the ego vehicle (every other vehicles positions are in the local coordinate system
    # of the ego vehicle)
    pos_ego = np.array([0,0,0])
    
    # Get the position of the current vehicle
    pos_vehicle = np.array(vehicle_position)

    # Calculate the angle between the vehicle and the ego vehicle
    angle_rad = np.arctan2(pos_vehicle[1] - pos_ego[1], pos_vehicle[0] - pos_ego[0])
    angle_deg = angle_rad * 180. / np.pi
    angle_deg = angle_deg % 360. # Normalize the angle to [0, 360]

    # Get the yaw angle (heading) of the vehicle
    vehicle_heading_angle_rad = vehicle_yaw
    vehicle_heading_angle_deg = vehicle_heading_angle_rad * 180 / np.pi
    vehicle_heading_angle_deg = vehicle_heading_angle_deg % 360. # Normalize the angle to [0, 360]

    other_vehicle_points_towards_ego = abs(vehicle_heading_angle_deg - angle_deg + 180) % 360 < threshold

    return other_vehicle_points_towards_ego, vehicle_heading_angle_deg

def get_vehicle_str(vehicle, neglect_pos=False):
    """
    :param vehicle (dict)
    :return important_object_str, vehicle_description, vehicle_location_description
    """
    # Determine the rough position of the vehicle relative to the ego (front, front-left, front-right)
    if not neglect_pos:
        rough_pos_str = f"{get_pos_str(vehicle['position'])} of the ego vehicle"

    # Determine the type of vehicle based on its type_id
    if 'firetruck' in vehicle['type_id']:
        vehicle_type = 'firetruck'
    elif 'police' in vehicle['type_id']:
        vehicle_type = 'police car'
    elif 'ambulance' in vehicle['type_id']:
        vehicle_type = 'ambulance'
    elif 'jeep' in vehicle['type_id']:
        vehicle_type = 'jeep'
    elif 'micro' in vehicle['type_id']:
        vehicle_type = 'small car'
    elif 'nissan.patrol' in vehicle['type_id']:
        vehicle_type = 'SUV'
    elif 'european_hgv' in vehicle['type_id']:
        vehicle_type = 'HGV'
    elif 'sprinter' in vehicle['type_id']:
        vehicle_type = 'sprinter'
    elif 'walker' in vehicle['type_id']:
        vehicle_type = 'pedestrian'
    else:
        vehicle_type = vehicle.get('base_type', 'vehicle')

    # Determine the color of the vehicle
    color_str = vehicle.get('color_name') + ' ' if vehicle.get('color_name') is not None \
                                                and vehicle.get('color_name') != 'None' else ''
    if vehicle.get('color') is not None and vehicle.get('color') != 'None':
        color_str = rgb_to_color_name(vehicle['color']) + ' '
        if vehicle['color'] == [0, 28, 0] or vehicle['color'] == [12, 42, 12]:
            color_str = 'dark green '
        elif vehicle['color'] == [211, 142, 0]:
            color_str = 'yellow '
        elif vehicle['color'] == [145, 255, 181]:
            color_str = 'blue '
        elif vehicle['color'] == [215, 88, 0]:
            color_str = 'orange '

    # Construct a string description of the vehicle
    if not neglect_pos:
        description = vehicle_type
        important_object_str = f'the {color_str}{description} {rough_pos_str}'
        vehicle_description = f'{color_str}{description}'
        vehicle_location_description = f'the {color_str}{description} that is {rough_pos_str}'
    else:
        description = vehicle_type
        important_object_str = f'the {color_str}{description}'
        vehicle_description = f'{color_str}{description}'
        vehicle_location_description = f'the {color_str}{description}'

    return important_object_str, vehicle_description, vehicle_location_description

def get_pedestrian_str(pedestrian):
    """
    :param pedestrian (dict)
    :return important_object_str
    """
    if -2 < pedestrian['position'][1] < 2:
        rough_pos_str = 'to the front of the ego vehicle'
    elif pedestrian['position'][1] > 2:
        rough_pos_str = 'to the front right of the ego vehicle'
    else:
        rough_pos_str = 'to the front left of the ego vehicle'

    return f'the pedestrian {rough_pos_str}'

def get_bicycle_str(vehicle):
    color_str = vehicle.get('color_name') + ' ' if vehicle.get('color_name') is not None \
                                                and vehicle.get('color_name') != 'None' else ''
    if vehicle.get('color') is not None and vehicle.get('color') != 'None':
        color_str = rgb_to_color_name(vehicle['color']) + ' '
        if vehicle['color'] == [0, 28, 0] or vehicle['color'] == [12, 42, 12]:
            color_str = 'dark green '
        elif vehicle['color'] == [211, 142, 0]:
            color_str = 'yellow '
        elif vehicle['color'] == [145, 255, 181]:
            color_str = 'blue '
        elif vehicle['color'] == [215, 88, 0]:
            color_str = 'orange '

    important_object_str = f'{color_str}bicycle'
    return important_object_str

def get_rel_lane_vehicles(vehicles, rel_lane, backwards=True):
    """
    :param other_vehicles (list of actor dict), be sure to have 'position' attribute
    :param the relative lane index, -1 for left lane, 1 for right lane (int)
    :param the direction we count, backward or forward
    :return list of vehicle dicts satisfy conditions, sorted by distance
    """

    filtered_vehicles = []
    for vehicle in vehicles:
        if vehicle['class'] == 'ego_vehicle':
            continue
        if vehicle['lane_relative_to_ego'] == rel_lane and \
            ((backwards == True and vehicle['position'][0] < 1.0) or \
             (backwards == False and vehicle['position'][0] > -1.0)):
            filtered_vehicles.append(vehicle)

    filtered_vehicles = sorted(filtered_vehicles, key=lambda actor: actor.get('distance', float('inf')))

    return filtered_vehicles

def get_clear_distance_of_lane(vehicles, rel_lane, backwards=True):
    """
    :param other_vehicles (list of actor dict), be sure to have 'position' attribute
    :param the relative lane index, -1 for left lane, 1 for right lane (int)
    :param the direction we count, backward or forward
    :return distance to closest vehicle of this lane
    """

    vehicle_list = get_rel_lane_vehicles(vehicles, rel_lane, backwards)

    if vehicle_list:
        return vehicle_list[0]['distance']
    return 99999

def get_vehicle_in_lane_within_threshold(vehicles, rel_lane, threshold=30.0, backwards=True):
    """
    :param other_vehicles (list of actor dict), be sure to have 'position' attribute
    :param the relative lane index, -1 for left lane, 1 for right lane (int)
    :param the direction we count, backward or forward
    :return all actor dicts of vehicle within given distance threshold
    """

    vehicle_list = get_rel_lane_vehicles(vehicles, rel_lane, backwards)

    if vehicle_list:
        return [x for x in vehicle_list if x['distance'] < threshold]
    return []

def get_project_camera_and_corners(obj_dict, cam_dict):
    """
    :param obj_dict (list of actor dict)
    :param self.CAM_DICT
    :return dicts of format {
                'cam_key': key, # camera name
                'corner_count': corner_count, # corner count in this camera
                below come from project_all_corners in graph_utils
                'projected_points': projected_points,
                'projected_points_meters': projected_points_meters,
            }
    """
    cam_keys = cam_dict.keys()
    project_list = []
    for key in cam_keys:
        res = get_vehicle_projected_corners(cam_dict[key], obj_dict)
        _, in_view_list = res
        projected_points, projected_points_meters = project_all_corners(obj_dict,
                                                                        cam_dict[key]['intrinsic'],
                                                                        cam_dict[key]['world2cam'])
        corner_count = sum(in_view_list)
        project_list.append(
            {
                'cam_key': key,
                'corner_count': corner_count,
                'projected_points': projected_points,
                'projected_points_meters': projected_points_meters,
            }
        )
    project_list = sorted(project_list, key=lambda x: x['corner_count'], reverse=True)
    project_list = [x for x in project_list if x['corner_count'] > 0]
    return project_list

def is_in_tunnel(x, y, town_name, road_id):
    """
    Determine if a vehicle is inside a tunnel based on its position and road information.

    Args:
        x (float): The x-coordinate of the vehicle.
        y (float): The y-coordinate of the vehicle.
        town_name (str): The name of the current town.
        road_id (int): The road ID where the vehicle is located.
        tunnels (list): A list of dictionaries containing tunnel information.

    Returns:
        bool: True if the vehicle is in a tunnel, False otherwise.
    """
    for tunnel in TUNNELS:
        if (
            tunnel["town"] == town_name
            and tunnel["road_id"] == road_id
            and min(tunnel["x1"], tunnel["x2"]) <= x <= max(tunnel["x1"], tunnel["x2"])
            and min(tunnel["y1"], tunnel["y2"]) <= y <= max(tunnel["y1"], tunnel["y2"])
        ):
            return True
    return False

def calculate_angle_between_two_vehicles(vehicle1, vehicle2):
    angle1 = vehicle1['rotation'][2] if vehicle1['rotation'][2] >= 0.0 else vehicle1['rotation'][2] + 360.0
    angle2 = vehicle2['rotation'][2] if vehicle2['rotation'][2] >= 0.0 else vehicle2['rotation'][2] + 360.0
    return abs(angle1 - angle2)

def get_lane_curvature(carla_map, current_location, dist, backward=False):
    """
    returns: 'left' 'right' 'none' 'straight' 'junction'
    """
    dist_int = int(dist)
    curve_flag = False
    curve_str = 'straight'
    for i in range(4, dist_int, 2):
        curve_flag, curve_str = calculate_curvature(carla_map, current_location, float(i), backward)
        if curve_flag:
            return curve_str
    return curve_str

def calculate_curvature(carla_map, current_location, dist, backward=False):
    """
    returns: 'left' 'right' 'none' 'straight' 'junction'
    """
    current_location = carla.Location(x=current_location[0], y=current_location[1], z=current_location[2])
    waypoint = carla_map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if not waypoint:
        return False, 'none'
    
    initial_forward_vector = waypoint.transform.get_forward_vector()
    
    step = dist
    next_waypoint = waypoint.next(step) if not backward else waypoint.previous(step)
    
    if not next_waypoint:
        return False, 'none'
    
    next_waypoint = next_waypoint[0]
    
    if next_waypoint.is_junction:
        return False, 'junction'
    
    next_forward_vector = next_waypoint.transform.get_forward_vector()
    
    dot_product = initial_forward_vector.x * next_forward_vector.x + initial_forward_vector.y * next_forward_vector.y
    determinant = initial_forward_vector.x * next_forward_vector.y - initial_forward_vector.y * next_forward_vector.x
    angle = math.atan2(determinant, dot_product)
    
    if angle < -CURVATURE_TRIGGER_VALUE:
        return True, 'left'
    elif angle > CURVATURE_TRIGGER_VALUE:
        return True, 'right'
    else:
        return False, 'straight'

def get_command_int_by_current_measurement(current_measurement, ego_vehicle):
    """
    Get current effective command int by measurement and ego_vehicle dict.

    Args:
        current_measurement (dict): The current measurement dict.
        ego_vehicle (dict): The dict of ego vehicle.

    Returns:
        command_int (int): The current command int.
    """
    command_int = current_measurement['next_command']
    x = (current_measurement['target_point'][0] - current_measurement['x'])**2
    y = (current_measurement['target_point'][1] - current_measurement['y'])**2
    command_distance = np.sqrt(x + y) # [debug]
    if current_measurement['command'] != 4 or (ego_vehicle['is_in_junction'] and command_int == 4):
        command_int = current_measurement['command']
        x = (current_measurement['x_command_near'] - current_measurement['x'])**2
        y = (current_measurement['y_command_near'] - current_measurement['y'])**2
    command_distance = np.sqrt(x + y)
    if command_distance > 30:
        command_int = 4
    if command_int == 5 and ego_vehicle['lane_change'] in [0, 1]: # cannot change to the left lane
        command_int = 4
    if command_int == 6 and ego_vehicle['lane_change'] in [0, 2]: # cannot change to the right lane
        command_int = 4

    return command_int

def get_vehicle_approach_approx(vehicle):
    speed = vehicle.get('speed', 4.0)
    distance = vehicle['distance']
    if -0.01 < speed < 0.01: speed = 0.01
    return distance / speed

def determine_basic_vector_crossing(command_int, other_vehicle, whitelist={}, following_list=[], first_appear_history={}):
    """
    Judge vehicle crossing by simple vector checks.
    
    Args:
        command_int (dict): Get current command.
        other_vehicle (dict): Other vehicle's information dictionary.

    Returns:
        crossed (bool): True if path crosses
        reason (str): Reason why path crosses
        action (str): Action may lead to collision
        must_stop (bool): Must stop flag
    """      

    basic_intersect_flag = False
    reason = None
    action = None
    must_stop_flag = False
    in_whitelist = other_vehicle['id'] in whitelist and other_vehicle['approaching_dot_product'] < 0
    acc_dist = INF_MAX
    for key in whitelist:
        if whitelist[key] < acc_dist:
            acc_dist = whitelist[key]

    history_straight_threshold = INF_MAX
    history_left_threshold = INF_MAX
    history_right_threshold = INF_MAX

    history_length = len(first_appear_history)

    if history_length >= ADJUST_INTERVAL_COUNT:
        recent_keys = list(islice(first_appear_history.keys(), max(0, history_length - ADJUST_INTERVAL_COUNT), None))

        straight_deltas = []
        left_deltas     = []
        right_deltas    = []

        for key in recent_keys:
            straight, left, right = first_appear_history[key]
            if straight is not None: straight_deltas.append(straight)
            if left     is not None: left_deltas.append(left)
            if right    is not None: right_deltas.append(right)

        front_adjust_min = RIGHT_ADJUST_MIN if command_int in [2] else FRONT_ADJUST_MIN
        front_adjust_down_margin = RIGHT_ADJUST_DOWN_MARGIN if command_int in [2] else FRONT_ADJUST_DOWN_MARGIN
        if straight_deltas:
            history_straight_threshold = max(max(straight_deltas) - front_adjust_down_margin, front_adjust_min)
        if left_deltas:
            history_left_threshold     = max(max(left_deltas)     - LEFT_ADJUST_DOWN_MARGIN,  LEFT_ADJUST_MIN)
        if right_deltas:
            history_right_threshold    = max(max(right_deltas)    - RIGHT_ADJUST_DOWN_MARGIN, RIGHT_ADJUST_MIN)

    
    print_debug(f"[debug] vehicle {other_vehicle['id']}'s in white list: {in_whitelist}. whitelist: {whitelist}. acc_dist = {acc_dist}. history_straight_threshold, history_left_threshold, history_right_threshold = {history_straight_threshold, history_left_threshold, history_right_threshold}")

    straight_delta_time = None
    left_delta_time = None
    right_delta_time = None

    if 'base_type' not in other_vehicle and 'walker' not in other_vehicle['type_id']:
        # sometimes this bug occurs
        print(f"Warning: in determine_basic_vector_crossing, vehicle id={other_vehicle['id']} is passed but its information is incomplete (no base type).")
        return False, None, None, False, (straight_delta_time, left_delta_time, right_delta_time)
    
    if other_vehicle['id'] in following_list or \
       (abs(other_vehicle['position'][1]) < 1.5 and other_vehicle['position'][0] < -2.0):
        # same lane and at rear
        return False, None, None, False, (straight_delta_time, left_delta_time, right_delta_time)

    if 'num_points' in other_vehicle and other_vehicle['num_points'] < 3:
        # doesn't appear
        return False, None, None, False, (straight_delta_time, left_delta_time, right_delta_time)
    
    raw_ratio = INTERSECTION_CROSS_RATIO if command_int != 2 else RIGHT_TURN_CROSS_RATIO
    straight_cross_ratio = raw_ratio if command_int not in [1, 2] else raw_ratio / TURNING_STRAIGHT_TIGHTEN_RATIO
    straight_dist_thres = INTERSECTION_DISTANCE_THRESHOLD if command_int not in [1, 2] else INTERSECTION_DISTANCE_THRESHOLD / TURNING_STRAIGHT_TIGHTEN_RATIO
    cross_interval = INTERSECTION_CROSS_INTERVAL if command_int in [1, 3] else RIGHT_TURN_INTERSECTION_CROSS_INTERVAL
    
    # when right and straight, check front
    if command_int in [2, 3]:
        print_debug(f"[debug] other_vehicle['intersection_point'] = {other_vehicle['intersection_point']}, other_vehicle['actor_arrive_intersection_time'] = {other_vehicle['actor_arrive_intersection_time']}, other_vehicle['ego_to_intersection'] = {other_vehicle['ego_to_intersection']}, straight_dist_thres = {straight_dist_thres}")
        if other_vehicle['intersection_point'] is not None and \
        other_vehicle['actor_arrive_intersection_time'] > 0.0 and \
        0.0 <= other_vehicle['ego_to_intersection'] <= straight_dist_thres:
            straight_delta_time = other_vehicle['actor_arrive_intersection_time'] - other_vehicle['ego_to_intersection'] / INTERSECTION_EGO_DEFAULT_SPEED
            intersection_cross_interval = min(straight_cross_ratio * (other_vehicle['ego_to_intersection'] / INTERSECTION_EGO_DEFAULT_SPEED) ** 2, cross_interval)
            print_debug(f"[debug] straight id = {other_vehicle['id']}, delta_time = {straight_delta_time}, intersection_cross_interval = {intersection_cross_interval}") #
            if straight_delta_time < INTERSECTION_CONSIDER_INTERVAL:
                print_debug(f"[debug] id = {other_vehicle['id']} crosses path at front!") #
                basic_intersect_flag = True
                if straight_delta_time < min(intersection_cross_interval, history_straight_threshold) and not in_whitelist and other_vehicle['distance'] < acc_dist:
                    reason = "is crossing the ego vehicle's path to the front"
                    action = "drives forward"
                    must_stop_flag = True
                else:
                    reason = "is going to cross the ego vehicle's path to the front in the future"
                    action = "drives too slow"
        print_debug(f"[debug] straight_delta_time = {straight_delta_time}") #
    
    if command_int in [1]:
        if other_vehicle['intersection_point_left'] is not None and \
           other_vehicle['actor_arrive_intersection_time_left'] > 0.0 and \
           0.0 <= other_vehicle['ego_to_intersection_left'] <= INTERSECTION_DISTANCE_THRESHOLD:
            left_delta_time = other_vehicle['actor_arrive_intersection_time_left'] - other_vehicle['ego_to_intersection_left'] / INTERSECTION_EGO_DEFAULT_SPEED
            intersection_cross_interval = min(raw_ratio * (other_vehicle['ego_to_intersection_left'] / INTERSECTION_EGO_DEFAULT_SPEED) ** 2, cross_interval)
            # print(f"[debug] left id = {other_vehicle['id']}, delta_time = {left_delta_time}, intersection_cross_interval = {intersection_cross_interval}")
            # print(f"[debug] id = {other_vehicle['id']} delta_time = {delta_time}") #
            if left_delta_time < INTERSECTION_CONSIDER_INTERVAL:
                # print(f"[debug] id = {other_vehicle['id']} crosses path at left!") #
                basic_intersect_flag = True
                if left_delta_time < min(intersection_cross_interval, history_left_threshold) and not in_whitelist and other_vehicle['distance'] < acc_dist:
                    reason = "is crossing the ego vehicle's path to the left"
                    action = "continue turning left"
                    must_stop_flag = True
                elif not must_stop_flag:
                    reason = "is going to cross the ego vehicle's path to the left in the future"
                    action = "turns left but drives too slow"
    
    # if command_int in [2]: # no need to check right
    if False:
        if other_vehicle['intersection_point_right'] is not None and \
           other_vehicle['actor_arrive_intersection_time_right'] > 0.0 and \
           0.0 <= other_vehicle['ego_to_intersection_right'] <= INTERSECTION_DISTANCE_THRESHOLD:
            right_delta_time = other_vehicle['actor_arrive_intersection_time_right'] - other_vehicle['ego_to_intersection_right'] / INTERSECTION_EGO_DEFAULT_SPEED
            intersection_cross_interval = min(raw_ratio * (other_vehicle['ego_to_intersection_right'] / INTERSECTION_EGO_DEFAULT_SPEED) ** 2, cross_interval)
            # print(f"[debug] right id = {other_vehicle['id']}, delta_time = {right_delta_time}, intersection_cross_interval = {intersection_cross_interval}")
            # print(f"[debug] id = {other_vehicle['id']} delta_time = {delta_time}") #
            if right_delta_time < INTERSECTION_CONSIDER_INTERVAL:
                # print(f"[debug] id = {other_vehicle['id']} crosses path at right!") #
                basic_intersect_flag = True
                if right_delta_time < min(intersection_cross_interval, history_right_threshold) and not in_whitelist and other_vehicle['distance'] < acc_dist:
                    reason = "is crossing the ego vehicle's path to the right"
                    action = "continue turning right"
                    must_stop_flag = True
                elif not must_stop_flag:
                    reason = "is going to cross the ego vehicle's path to the right in the future"
                    action = "turns right but drives too slow"
    
    print_debug(f"[debug] basic_intersect_flag, reason, action, must_stop_flag = {basic_intersect_flag, reason, action, must_stop_flag}, tuple = {(straight_delta_time, left_delta_time, right_delta_time)}")
    return basic_intersect_flag, reason, action, must_stop_flag, (straight_delta_time, left_delta_time, right_delta_time)

def point_inside_boundingbox(point, bb_center, bb_extent, multiplier=1.2):
    """Checks whether or not a point is inside a bounding box."""
    # copied from atomic criteria from carla.

    # pylint: disable=invalid-name
    A = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
    B = carla.Vector2D(bb_center.x + multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
    D = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y + multiplier * bb_extent.y)
    M = carla.Vector2D(point.x, point.y)

    AB = B - A
    AD = D - A
    AM = M - A
    am_ab = AM.x * AB.x + AM.y * AB.y
    ab_ab = AB.x * AB.x + AB.y * AB.y
    am_ad = AM.x * AD.x + AM.y * AD.y
    ad_ad = AD.x * AD.x + AD.y * AD.y

    return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad  # pylint: disable=chained-comparison

def do_ego_must_stop(ego_location, stop_sign_dict, map):
    """
    Check if the given actor is in the affecting range of the stop.
    Without using waypoints, a stop might not be detected if the actor is moving at the lane edge.
    """
    # copied from atomic criteria from carla.
    # Quick distance test
    PROXIMITY_THRESHOLD = 4.0 # same as carla
    stop_location = stop_sign_dict['trigger_volume_location']
    stop_location = carla.Location(x=stop_location[0],
                                   y=stop_location[1],
                                   z=stop_location[2])
    actor_location = carla.Location(x=ego_location[0],
                                    y=ego_location[1],
                                    z=ego_location[2])
    if stop_location.distance(actor_location) > PROXIMITY_THRESHOLD:
        return False

    # Check if the any of the actor wps is inside the stop's bounding box.
    # Using more than one waypoint removes issues with small trigger volumes and backwards movement
    stop_extent = stop_sign_dict['trigger_volume_extent']
    stop_extent = carla.Vector3D(x=stop_extent[0],
                                 y=stop_extent[1],
                                 z=stop_extent[2])
    wp = map.get_waypoint(actor_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if point_inside_boundingbox(wp.transform.location, stop_location, stop_extent):
        return True

    return False

def get_lateral_distance_to_lane_center(carla_map, location):
    """
    Returns lateral distance to vehicle's lane center.
    Right is positive while left is negative.
    """

    vehicle_location = carla.Location(x=location[0],
                                      y=location[1],
                                      z=location[2])
    
    waypoint = carla_map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    lane_location = waypoint.transform.location
    lane_right_vector = waypoint.transform.get_right_vector()
    
    dx = vehicle_location.x - lane_location.x
    dy = vehicle_location.y - lane_location.y
    offset_vector = np.array([dx, dy])
    
    right_vec = np.array([lane_right_vector.x, lane_right_vector.y])
    
    lateral_distance = np.dot(offset_vector, right_vec)
    
    return lateral_distance