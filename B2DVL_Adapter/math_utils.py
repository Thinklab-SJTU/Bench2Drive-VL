import math
import numpy as np
import re
from collections import defaultdict
import string
from shapely.geometry import Polygon

def dcg(scores):
    return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores))

def ndcg(gt_list, vlma_list, weights):

    vlma_scores = [weights[obj] for obj in vlma_list if (obj in weights) and (obj in gt_list)]
    ideal_scores = [weights[obj] for obj in gt_list if (obj in weights) and (obj in vlma_list)]
    
    dcg_val = dcg(vlma_scores)
    idcg_val = dcg(ideal_scores)
    
    return dcg_val / idcg_val if idcg_val > 0 else 0

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
    Transform a point in ego_vehicle coordinates to world coordinates.

    Params:
        ego_coordinates (list or tuple): The (x', y', z') ego vehicle coordinates.
        world2ego_matrix (list): The 4x4 world-to-ego transformation matrix.

    Returns:
        tuple: The transformed (x, y, z) coordinates in world coordinate system.
    """

    P_ego = np.array(list(ego_coordinates) + [1])
    M_ego2world = np.linalg.inv(np.array(world2ego_matrix))

    P_world_homogeneous = M_ego2world @ P_ego

    return tuple(P_world_homogeneous[:3])

def calculate_yaw(start, end):
    """
    Calculate the yaw angle (in degrees) from the start point to the end point.
    Only the x and y components of the points are considered.

    Params:
        start (tuple or list): The (x, y) coordinates of the start point.
        end (tuple or list): The (x, y) coordinates of the end point.

    Returns:
        float: The yaw angle in degrees, with 0 degrees along the x-axis and 90 degrees along the y-axis.
    """

    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]

    yaw_radians = np.arctan2(delta_y, delta_x)

    yaw_degrees = np.degrees(yaw_radians)

    return normalize_yaw_degrees(yaw_degrees)

def normalize_yaw_degrees(yaw_degrees):
    if yaw_degrees < 0:
        yaw_degrees += 360

    return yaw_degrees

def get_mean_angle(angle1, angle2):
    """
    Given two angles, angle1 and angle2, this function finds the closest angle
    from the set [angle2 - 360, angle2 - 180, angle2, angle2 + 180], 
    and returns the average of angle1 and the closest angle.

    Params:
        angle1 (float): The first angle.
        angle2 (float): The second angle.

    Returns:
        float: The average of angle1 and the closest adjusted angle.
    """
    
    angle1 = angle1 % 360
    angle2 = angle2 % 360

    candidates = [angle2 - 360, angle2 - 180, angle2, angle2 + 180]

    closest_angle = min(candidates, key=lambda x: abs(x - angle1))

    return (angle1 + closest_angle) / 2

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in a 2D plane (x, y).
    
    Params:
        point1 (tuple): The (x, y) coordinates of the first point.
        point2 (tuple): The (x, y) coordinates of the second point.
        
    Returns:
        float: The distance between the two points.
    """
    
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    distance = math.sqrt(dx**2 + dy**2)
    
    return distance

def rotate_point(x, y, yaw):
    """
    Rotate a point (x, y) by the given yaw (in degrees).
    
    Params:
        x (float): The x coordinate of the point.
        y (float): The y coordinate of the point.
        yaw (float): The rotation angle in degrees.
        
    Returns:
        (float, float): The rotated coordinates (x', y').
    """
    theta = math.radians(yaw)
    
    x_rot = x * math.cos(theta) - y * math.sin(theta)
    y_rot = x * math.sin(theta) + y * math.cos(theta)
    
    return x_rot, y_rot

def calculate_rectangle_corners(center_x, center_y, yaw, extent_x, extent_y):
    """
    Calculate the four corners of a rotated rectangle.

    Params:
        center_x (float): The x coordinate of the center of the rectangle.
        center_y (float): The y coordinate of the center of the rectangle.
        yaw (float): The rotation angle in degrees (counter-clockwise from x-axis).
        extent_x (float): The half-width of the rectangle along the x-axis (horizontal).
        extent_y (float): The half-height of the rectangle along the y-axis (vertical).
        
    Returns:
        list of tuples: A list containing the coordinates of the four corners in order (top-left, top-right, bottom-right, bottom-left).
    """
    
    corners_relative = [
        (-extent_x, extent_y),  # top-left
        (extent_x, extent_y),   # top-right
        (extent_x, -extent_y),  # bottom-right
        (-extent_x, -extent_y)  # bottom-left
    ]
    
    corners_rotated = []
    for corner in corners_relative:
        rotated_x, rotated_y = rotate_point(corner[0], corner[1], yaw)
        corners_rotated.append((center_x + rotated_x, center_y + rotated_y))
    
    return corners_rotated

def is_intersecting_and_iou(rectangleA, rectangleB):
    """   
    :param rectangleA: vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param rectangleB: vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :return: bool: intersect or not; iou: float
    """
    
    polygonA = Polygon(rectangleA)
    polygonB = Polygon(rectangleB)
    
    intersection = polygonA.intersection(polygonB)
    union = polygonA.union(polygonB)
    
    is_intersect = intersection.is_empty is False
    
    intersection_area = intersection.area
    union_area = union.area
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return is_intersect, iou

def clean_key(desc):
    desc = desc.strip().strip(string.punctuation + " ")
    if desc:
        desc = desc[0].upper() + desc[1:]
    return desc

def parse_objects(text):
    """
    Parses well-formed object(tag) expressions from text.
    Returns a dict mapping cleaned object descriptions to their tags.
    """
    pattern = re.compile(r'([^(]+?)\(([^()]+?)\)')
    obj_dict = {}
    obj_count = defaultdict(int)

    for match in pattern.finditer(text):
        raw_desc = match.group(1)
        tag = match.group(2).strip()
        desc = clean_key(raw_desc)
        obj_count[desc] += 1
        if obj_count[desc] > 1:
            desc = f"{desc} {obj_count[desc]}"
        obj_dict[desc] = tag

    return obj_dict

if __name__ == "__main__":
    test_str = "The construction warning sign(<c14757<CAM_FRONT,1095.6,496.6>>), the black car that is to the front of the ego vehicle(<c14785<CAM_FRONT,800.0,513.9>>)."
    test_str2 = "The black car that is to the front of the ego vehicle(<c14785<CAM_FRONT,800.0,533.2>>), because it has stopped right to the front of the ego vehicle, and the collision will happen if the ego vehicle drives forward along the current lane."
    print(parse_objects(test_str))
    print(parse_objects(test_str2))