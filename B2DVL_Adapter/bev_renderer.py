import numpy as np
import cv2
import carla
from generator_modules import get_vehicle_str, transform_to_ego_coordinates
import math
import os

MINIMAL = int(os.environ.get("MINIMAL", 0))

# Define colors in BGR format
COLOR_OFF        = (150, 150, 150)
COLOR_BORDER     = (50, 50, 50)

COLOR_RED        = (0, 0, 255)
COLOR_GREEN      = (0, 255, 0)
COLOR_BLUE       = (255, 0, 0)

COLOR_YELLOW     = (0, 255, 255)
COLOR_CYAN       = (255, 255, 0)
COLOR_MAGENTA    = (255, 0, 255)

COLOR_WHITE      = (255, 255, 255)
COLOR_BLACK      = (0, 0, 0)
COLOR_GRAY       = (128, 128, 128)
COLOR_DARK_GRAY  = (64, 64, 64)
COLOR_LIGHT_GRAY = (200, 200, 200)

COLOR_ORANGE     = (0, 165, 255)
COLOR_PURPLE     = (128, 0, 128)
COLOR_BROWN      = (19, 69, 139)
COLOR_PINK       = (203, 192, 255)
COLOR_LIME       = (0, 255, 0)
COLOR_NAVY       = (128, 0, 0)
COLOR_TEAL       = (128, 128, 0)

COLOR_BUTTER_0 = (79, 233, 252)
COLOR_BUTTER_1 = (0, 212, 237)
COLOR_BUTTER_2 = (0, 160, 196)

COLOR_ORANGE_0 = (62, 175, 252)
COLOR_ORANGE_1 = (0, 121, 245)
COLOR_ORANGE_2 = (0, 92, 209)

COLOR_CHOCOLATE_0 = (110, 185, 233)
COLOR_CHOCOLATE_1 = (17, 125, 193)
COLOR_CHOCOLATE_2 = (2, 89, 143)

COLOR_CHAMELEON_0 = (52, 226, 138)
COLOR_CHAMELEON_1 = (22, 210, 115)
COLOR_CHAMELEON_2 = (6, 154, 78)

COLOR_SKY_BLUE_0 = (207, 159, 114)
COLOR_SKY_BLUE_1 = (164, 101, 52)
COLOR_SKY_BLUE_2 = (135, 74, 32)

COLOR_PLUM_0 = (168, 127, 173)
COLOR_PLUM_1 = (123, 80, 117)
COLOR_PLUM_2 = (102, 53, 92)

COLOR_SCARLET_RED_0 = (41, 41, 239)
COLOR_SCARLET_RED_1 = (0, 0, 204)
COLOR_SCARLET_RED_2 = (0, 0, 164)

COLOR_ALUMINIUM_0 = (236, 238, 238)
COLOR_ALUMINIUM_1 = (207, 215, 211)
COLOR_ALUMINIUM_2 = (182, 189, 186)
COLOR_ALUMINIUM_3 = (133, 138, 136)
COLOR_ALUMINIUM_4 = (83, 87, 85)
COLOR_ALUMINIUM_4_5 = (64, 62, 66)
COLOR_ALUMINIUM_5 = (54, 52, 46)

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

CONSIDER_DISTANCE = 75
ANNO_CONSIDER_DISTANCE = 50
CONSIDER_Z = 30 # avoid the situation that the role actor is undergound


def project_point(world_point, K, extrinsics):
    """
    Projects a 3D world coordinate point onto the image plane.

    Args:
        world_point (list or np.ndarray): 3D world coordinate [x, y, z]
        K (np.ndarray): 3x3 camera intrinsic matrix
        extrinsics (np.ndarray): 4x4 camera extrinsic matrix

    Returns:
        np.ndarray: 2D image coordinate [u, v], or None if the point is behind the camera.
    """
    if isinstance(world_point, carla.Location):
        world_point = [world_point.x, world_point.y, world_point.z]
    K = np.array(K)
    extrinsics = np.array(extrinsics)
    
    # Convert to homogeneous coordinates (x, y, z, 1)
    world_point_h = np.append(world_point, 1)  

    # Transform world coordinates to camera coordinates
    camera_point = extrinsics @ world_point_h
    camera_point = camera_point[:3]  # Take the first three components (x, y, z)

    # Convert to OpenCV coordinate format (y, -z, x)
    pos_3d = np.array([camera_point[1], -camera_point[2], camera_point[0]])
    # Skip projection if the point is behind
    if pos_3d[2] <= 0:
        return None

    # Project to 2D image plane
    rvec = np.zeros((3, 1), np.float32)  # Rotation vector
    tvec = np.zeros((3, 1), np.float32)  # Translation vector
    dist_coeffs = np.zeros((5, 1), np.float32)  # Assume no distortion
    
    points_2d, _ = cv2.projectPoints(
        pos_3d, rvec, tvec, K, dist_coeffs
    )

    return [int(points_2d[0][0][0]), int(points_2d[0][0][1])]  # Return [u, v] image coordinates

def draw_traffic_light(cv_img, position, state, size=60):
    """
    Draw a traffic light icon on an image using OpenCV.

    Parameters:
    - cv_img: numpy array (H, W, 3), the image to draw on
    - position: (x, y) tuple, where to place the traffic light icon (centered)
    - state: str, one of ["Red", "Yellow", "Green", "Off", "Unknown"]
    - size: int, total height of the traffic light icon
    """

    try:
        # Calculate width/height and positions
        w = int(size / 2)
        h = size
        top_left = (position[0] - w//2, position[1] - h//2)
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw the background rectangle
        cv2.rectangle(cv_img, top_left, bottom_right, COLOR_BORDER, thickness=-1)

        # Circle positions (centered vertically)
        cx = top_left[0] + w // 2
        cy = top_left[1] + h // 6
        radius = int(w * 0.3)

        states = {
            "Red":   [COLOR_RED, COLOR_OFF, COLOR_OFF],
            "Yellow":[COLOR_OFF, COLOR_YELLOW, COLOR_OFF],
            "Green": [COLOR_OFF, COLOR_OFF, COLOR_GREEN],
            "Off":   [COLOR_OFF, COLOR_OFF, COLOR_OFF],
            "Unknown":[COLOR_OFF, COLOR_OFF, COLOR_OFF],
        }

        colors = states.get(state, [COLOR_OFF, COLOR_OFF, COLOR_OFF])

        for i in range(3):
            circle_y = cy + i * h // 3
            cv2.circle(cv_img, (cx, circle_y), radius, colors[i], thickness=-1)
    except Exception as e:
        print(f"BEV Renderer Draw Traffic Light: Exception occured, ignore this point: {e} ")

def draw_speed_signs(color_img, centers, limits, radius=30):
    """
    Draw speed limit signs on the given image.
    
    Parameters:
        color_img: np.ndarray, the target image (BGR).
        centers: list of (x, y) coordinates.
        limits: list of string values for speed limits.
        radius: outer circle radius in pixels.
    """
    try:
        white_radius = int(radius * 0.75)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = radius * 0.75 / 30
        thickness = 2

        for (x, y), limit in zip(centers, limits):
            center = (int(x), int(y))

            # Outer red circle
            cv2.circle(color_img, center, radius, (0, 0, 255), -1)  # Red (BGR)
            # Inner white circle
            cv2.circle(color_img, center, white_radius, (255, 255, 255), -1)  # White

            # Text size & draw
            text_size, _ = cv2.getTextSize(str(limit), font, font_scale, thickness)
            text_x = int(x - text_size[0] // 2)
            text_y = int(y + text_size[1] // 2)
            cv2.putText(color_img, str(limit), (text_x, text_y), font, font_scale, (50, 50, 50), thickness, cv2.LINE_AA)
    except Exception as e:
        print(f"BEV Renderer Draw Speed Sign: Exception occured, ignore this point: {e} ")

def draw_stop_sign(color_img, centers, size=25):
    """
    Draw stop signs (octagons with 'STOP') on an image.

    Parameters:
        color_img: np.ndarray, the image to draw on.
        centers: list of (x, y) coordinates.
        size: the radius-like half-diagonal of the stop sign.
    """
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = size / 45
        thickness = 2
        text_color = (255, 255, 255)  # white
        border_color = (255, 255, 255)
        fill_color = (0, 0, 255)  # red

        for (cx, cy) in centers:
            cx, cy = int(cx), int(cy)
            angle_step = 2 * np.pi / 8
            points = []
            for i in range(8):
                angle = angle_step * i + np.pi / 8  # Rotate so it's upright
                x = int(cx + size * np.cos(angle))
                y = int(cy + size * np.sin(angle))
                points.append((x, y))
            points = np.array([points], dtype=np.int32)

            # Optional white border (slightly larger octagon)
            border_pts = []
            border_offset = int(size * 0.1)
            for i in range(8):
                angle = angle_step * i + np.pi / 8
                x = int(cx + (size + border_offset) * np.cos(angle))
                y = int(cy + (size + border_offset) * np.sin(angle))
                border_pts.append((x, y))
            border_pts = np.array([border_pts], dtype=np.int32)

            cv2.fillPoly(color_img, border_pts, border_color)
            cv2.fillPoly(color_img, points, fill_color)

            # Draw STOP text
            text = "STOP"
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            cv2.putText(color_img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    except Exception as e:
        print(f"BEV Renderer Draw Stop Sign: Exception occured, ignore this point: {e} ")

def draw_labeled_dot(img, x, y, label, 
                     color=(255, 255, 255), 
                     font_scale=1.0, 
                     radius=10,
                     line_spacing=5):
    """
    Draws a dot at a given position with a text label on the top-right.

    Args:
        img (np.ndarray): Image to draw on.
        x, y (int): Center coordinates of the dot.
        label (str): Text label to display.
        color (tuple): BGR color of the dot and text (default: white).
        font_scale (float): Scale factor for text size.
        radius (int): Radius of the dot (default: 10).
        line_spacing (int): Gap height between lines (default: 5).
    """
    # print(f"[debug] labeled dot, type(img) = {type(img)}, shape is {img.shape}, type is {img.dtype}")
    x, y = int(x), int(y)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        # Draw the circle
        cv2.circle(img, (x, y), radius, color, -1)

        # Calculate text size and position offset
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x + radius + 2
        base_text_y = y - radius - 2

        lines = label.split('\n')
        for i, line in enumerate(lines):
            text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_y = base_text_y + i * (text_size[1] + line_spacing)

            # Avoid drawing above the image
            if text_y < 0:
                text_y = y + radius + (i + 1) * (text_size[1] + line_spacing)

            cv2.putText(img, line, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    except Exception as e:
        print(f"BEV Renderer Draw Labeled Dot: Exception occured, ignore this point: {e} ")

def draw_dot_list(img, dot_list,
                  color=(255, 255, 255),
                  radius=5):
    """
    Draws a dots on the image.

    Args:
        img (np.ndarray): Image to draw on.
        dot_list (list): 2D coordinate list, given in [[x1, y1], [x2, y2], ... [xn, yn]]
        color (tuple): BGR color of the dot and text (default: white).
        radius (int): Radius of the dot (default: 10).
    """
    # print(f"[debug] labeled dot, type(img) = {type(img)}, shape is {img.shape}, type is {img.dtype}")
    for dot in dot_list:
        try:
            x, y = int(dot[0]), int(dot[1])
            thickness = 2
            cv2.circle(img, (x, y), radius, color, -1)
        except Exception as e:
            print(f"BEV Renderer Draw Dot List: Exception occured, ignore this point: {e} ")

def draw_fading_polyline(
    image, 
    points, 
    color=(255, 255, 255), 
    thickness=16
):
    """
    Draws a fading polyline with decreasing opacity between points.

    Parameters:
        image: np.ndarray - the BGR image to draw on.
        points: list of (x, y) - at least 2 points.
        color: tuple of BGR - default is white.
        thickness: int - line thickness.
    """
    # Check
    if len(points) < 2:
        return

    # Make sure all points are integer tuples
    points = [(int(x), int(y)) for x, y in points]

    # Convert to float image for blending
    overlay = image.copy()

    # Define fading alpha values
    num_segments = len(points) - 1
    alphas = np.linspace(0.8, 0.2, num_segments)

    for i in range(num_segments):
        pt1 = points[i]
        pt2 = points[i + 1]
        segment_layer = overlay.copy()
        try:
            cv2.line(segment_layer, pt1, pt2, color, thickness, cv2.LINE_AA)
            cv2.addWeighted(segment_layer, alphas[i], overlay, 1 - alphas[i], 0, overlay)
        except Exception as e:
            print(f"BEV Renderer Polyline Drawer: Exception occured, ignore this point: {e} ")
            break

    # Write the result back to the image
    image[:] = overlay

def cut_to_square(img):
    """
    Center-crops an image to the largest possible square.

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Square-cropped image.
    """
    h, w = img.shape[:2]
    size = min(h, w)

    # Compute top-left corner for cropping
    top = (h - size) // 2
    left = (w - size) // 2

    return img[top:top+size, left:left+size]

class RoadCameraProjector:
    def __init__(self, intrinsic_matrix, extrinsic_matrix, ego_location, precision=0.05, consider_radius=90.0, road_filter_radius=128.0):
        self.K = intrinsic_matrix
        self.RT = extrinsic_matrix
        self.precision = precision
        self.consider_radius = consider_radius
        self.road_filter_radius = road_filter_radius
        self.ego_location = ego_location

    def distance_to_ego_2d(self, location):
        if isinstance(location, carla.Location):
            location = [location.x, location.y, location.z]
        return math.sqrt((self.ego_location[0] - location[0])**2 + (self.ego_location[1] - location[1])**2)
    
    def waypoint_is_near(self, wp):
        return self.distance_to_ego_2d(wp.transform.location) < self.consider_radius

    def road_end_is_considered(self, wp):
        return self.distance_to_ego_2d(wp.transform.location) < self.road_filter_radius

    def lane_marking_color_to_bgr(self, lane_marking_color):
        """Maps the lane marking color enum specified in PythonAPI to a BGR color"""
        if lane_marking_color == carla.LaneMarkingColor.White:
            return COLOR_ALUMINIUM_2  # COLOR_ALUMINIUM_2
        elif lane_marking_color == carla.LaneMarkingColor.Blue:
            return COLOR_SCARLET_RED_2  # COLOR_SKY_BLUE_0
        elif lane_marking_color == carla.LaneMarkingColor.Green:
            return COLOR_CHAMELEON_0   # COLOR_CHAMELEON_0
        elif lane_marking_color == carla.LaneMarkingColor.Red:
            return COLOR_SCARLET_RED_0    # COLOR_SCARLET_RED_0
        elif lane_marking_color == carla.LaneMarkingColor.Yellow:
            return COLOR_ORANGE_0   # COLOR_ORANGE_0
        return (0, 0, 0)  # default to black

    def draw_solid_line(self, img, color, closed, points, width):
        if len(points) >= 2:
            cv2.polylines(img, [np.array(points, dtype=np.int32)], closed, color, width)

    def draw_broken_line(self, img, color, closed, points, width):
        grouped = [points[i:i + 20] for i in range(0, len(points), 20)]
        selected = [g for i, g in enumerate(grouped) if i % 3 == 0 and len(g) >= 2]
        for group in selected:
            cv2.polylines(img, [np.array(group, dtype=np.int32)], closed, color, width)

    def draw_lane(self, img, lane, color):
        for side in lane:
            lane_left = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
            lane_right = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]
            polygon = lane_left + list(reversed(lane_right))
            polygon = [self.world_to_pixel(p) for p in polygon]
            if len(polygon) > 2:
                cv2.polylines(img, [np.array(polygon, dtype=np.int32)], isClosed=True, color=color, thickness=5)
                cv2.fillPoly(img, [np.array(polygon, dtype=np.int32)], color=color)

    def get_lane_markings(self, lane_marking_type, lane_marking_color, waypoints, sign):
        margin = 0.25
        marking_1 = [self.world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]

        if lane_marking_type in [carla.LaneMarkingType.Broken, carla.LaneMarkingType.Solid]:
            return [(lane_marking_type, self.lane_marking_color_to_bgr(lane_marking_color), marking_1)]
        else:
            marking_2 = [self.world_to_pixel(lateral_shift(w.transform, sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]
            c = self.lane_marking_color_to_bgr(lane_marking_color)
            if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                return [(carla.LaneMarkingType.Broken, c, marking_1), (carla.LaneMarkingType.Solid, c, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                return [(carla.LaneMarkingType.Solid, c, marking_1), (carla.LaneMarkingType.Broken, c, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                return [(carla.LaneMarkingType.Broken, c, marking_1), (carla.LaneMarkingType.Broken, c, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                return [(carla.LaneMarkingType.Solid, c, marking_1), (carla.LaneMarkingType.Solid, c, marking_2)]
        return [(carla.LaneMarkingType.NONE, (0, 0, 0), [])]

    def draw_lane_marking_single_side(self, img, waypoints, sign):
        markings_list = []
        temp_waypoints = []
        current_lane_marking = carla.LaneMarkingType.NONE
        previous_marking_type = carla.LaneMarkingType.NONE
        previous_marking_color = carla.LaneMarkingColor.Other

        for sample in waypoints:
            lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking
            if lane_marking is None:
                continue

            marking_type = lane_marking.type
            marking_color = lane_marking.color

            if current_lane_marking != marking_type:
                markings = self.get_lane_markings(previous_marking_type, previous_marking_color, temp_waypoints, sign)
                markings_list.extend(markings)
                temp_waypoints = temp_waypoints[-1:]
                current_lane_marking = marking_type
            else:
                temp_waypoints.append(sample)
                previous_marking_type = marking_type
                previous_marking_color = marking_color

        markings = self.get_lane_markings(previous_marking_type, previous_marking_color, temp_waypoints, sign)
        markings_list.extend(markings)

        for marking in markings_list:
            if marking[0] == carla.LaneMarkingType.Solid:
                self.draw_solid_line(img, marking[1], False, marking[2], 2)
            elif marking[0] == carla.LaneMarkingType.Broken:
                self.draw_broken_line(img, marking[1], False, marking[2], 2)

    def draw_lane_marking(self, img, waypoints):
        self.draw_lane_marking_single_side(img, waypoints[0], -1)  # left
        self.draw_lane_marking_single_side(img, waypoints[1], 1)   # right

    def draw_arrow(self, img, transform, color=(238, 238, 236)):
        transform.rotation.yaw += 180
        forward = transform.get_forward_vector()
        transform.rotation.yaw += 90
        right_dir = transform.get_forward_vector()

        end = transform.location
        start = end - 2.0 * forward
        right = start + 0.8 * forward + 0.4 * right_dir
        left = start + 0.8 * forward - 0.4 * right_dir

        pts1 = [self.world_to_pixel(start), self.world_to_pixel(end)]
        pts2 = [self.world_to_pixel(left), self.world_to_pixel(start), self.world_to_pixel(right)]

        cv2.line(img, pts1[0], pts1[1], color, thickness=4)
        cv2.polylines(img, [np.array(pts2, dtype=np.int32)], isClosed=False, color=color, thickness=4)
    
    def draw_polygon(self, img, polygon, color, width=1):
        if not polygon or len(polygon) < 2:
            return

        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=color)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=width)

    def draw_topology(self, img, carla_topology, index):
        """ Draws traffic signs and the roads network with sidewalks, parking and shoulders by generating waypoints"""
        # print(f"[debug] BEV Renderer: 1, len(carla_topology) = {len(carla_topology)}")
        topology = [x for x in carla_topology if any(self.road_end_is_considered(wp) for wp in x)]
        # print(f"[debug] BEV Renderer: 2, len(topology) = {len(topology)}")
        topology = [x[index] for x in topology]
        # print(f"[debug] BEV Renderer: 3, len(topology) = {len(topology)}")
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []

        # print(f"[debug] BEV Renderer: start to draw topology")
        for waypoint in topology:
            # print(f"[debug] BEV Renderer: dealing with topology waypoint with location {waypoint.transform.location}")
            waypoints = []
            if self.waypoint_is_near(waypoint):
                waypoints.append(waypoint)

            # Generate waypoints of a road id. Stop when road id differs
            nxt = waypoint.next(self.precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    if self.waypoint_is_near(nxt):
                        waypoints.append(nxt)
                    nxt = nxt.next(self.precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            set_waypoints.append(waypoints)

            # Draw Shoulders, Parkings and Sidewalks
            PARKING_COLOR = COLOR_ALUMINIUM_4_5
            SHOULDER_COLOR = COLOR_ALUMINIUM_5
            SIDEWALK_COLOR = COLOR_ALUMINIUM_3

            shoulder = [[], []]
            parking = [[], []]
            sidewalk = [[], []]

            for w in waypoints:
                # Classify lane types until there are no waypoints by going left
                l = w.get_left_lane()
                while l and l.lane_type != carla.LaneType.Driving:
                    
                    if l.lane_type == carla.LaneType.Shoulder:
                        shoulder[0].append(l)

                    if l.lane_type == carla.LaneType.Parking:
                        parking[0].append(l)

                    if l.lane_type == carla.LaneType.Sidewalk:
                        sidewalk[0].append(l)

                    l = l.get_left_lane()

                # Classify lane types until there are no waypoints by going right
                r = w.get_right_lane()
                while r and r.lane_type != carla.LaneType.Driving:

                    if r.lane_type == carla.LaneType.Shoulder:
                        shoulder[1].append(r)

                    if r.lane_type == carla.LaneType.Parking:
                        parking[1].append(r)

                    if r.lane_type == carla.LaneType.Sidewalk:
                        sidewalk[1].append(r)

                    r = r.get_right_lane()

            # Draw classified lane types
            # print("[debug] BEV Renderer: draw_shoulder")
            self.draw_lane(img, shoulder, SHOULDER_COLOR)
            # print("[debug] BEV Renderer: draw_parking")
            self.draw_lane(img, parking, PARKING_COLOR)
            # print("[debug] BEV Renderer: draw_sidewalk")
            self.draw_lane(img, sidewalk, SIDEWALK_COLOR)

        # Draw Roads
        for waypoints in set_waypoints:
            if len(waypoints) > 0:
                waypoint = waypoints[0]
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [self.world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    self.draw_polygon(img, polygon, COLOR_ALUMINIUM_5, 5)
                    self.draw_polygon(img, polygon, COLOR_ALUMINIUM_5)

                # Draw Lane Markings and Arrows
                if not waypoint.is_junction:
                    self.draw_lane_marking(img, [waypoints, waypoints])
                    for n, wp in enumerate(waypoints):
                        if ((n + 1) % 400) == 0:
                            self.draw_arrow(img, wp.transform)
    
    def _render_walkers(self, img, walker):
        try:
            color = COLOR_CYAN

            bbx = walker['extent'][0]
            bby = walker['extent'][1]
            corners = [
                carla.Location(x=-bbx, y=-bby),
                carla.Location(x=bbx, y=-bby),
                carla.Location(x=bbx, y=bby),
                carla.Location(x=-bbx, y=bby)
            ]

            loc = carla.Location(x=walker['location'][0], y=walker['location'][1], z=walker['location'][2])
            rot = carla.Rotation(pitch=walker['rotation'][0], yaw=walker['rotation'][2], roll=walker['rotation'][1])
            walker_transform = carla.Transform(location=loc, rotation=rot)
            walker_transform.transform(corners)
            corners = [self.world_to_pixel(p) for p in corners]
            corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))

            cv2.fillPoly(img, [corners], color)
        except Exception as e:
            print(f"BEV Generator Warning: walker can not be rendered: {e}")
    
    def _render_vehicles(self, img, vehicle):
        try:
            if vehicle['class'] == 'ego_vehicle':
                color = COLOR_BLUE
            elif 'color' in vehicle:
                rgb = vehicle['color'].split(',')
                color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            else:
                color = COLOR_SKY_BLUE_0

            bbx = vehicle['extent'][0]
            bby = vehicle['extent'][1]
            corners = [
                carla.Location(x=-bbx, y=-bby),
                carla.Location(x=bbx - 0.8, y=-bby),
                carla.Location(x=bbx, y=0),
                carla.Location(x=bbx - 0.8, y=bby),
                carla.Location(x=-bbx, y=bby),
                carla.Location(x=-bbx, y=-bby)
            ]

            loc = carla.Location(x=vehicle['location'][0], y=vehicle['location'][1], z=vehicle['location'][2])
            rot = carla.Rotation(pitch=vehicle['rotation'][0], yaw=vehicle['rotation'][2], roll=vehicle['rotation'][1])
            vehicle_transform = carla.Transform(location=loc, rotation=rot)
            vehicle_transform.transform(corners)
            corners = [self.world_to_pixel(p) for p in corners]
            corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            thickness = 3

            cv2.fillPoly(img, [corners], color=color)
            cv2.polylines(img, [corners], isClosed=False, color=COLOR_SKY_BLUE_0, thickness=thickness)
        except Exception as e:
            print(f"BEV Generator Warning: vehicle can not be rendered: {e}")

    def world_to_pixel(self, location):
        uv = project_point([location.x, location.y, location.z], K=self.K, extrinsics=self.RT)
        u, v = uv[0], uv[1]
        return (u, v)

        world_coord = np.array([[location.x], [location.y], [location.z], [1.0]])

        cam_coord = self.RT @ world_coord
        cam_coord = cam_coord[:3]

        pixel_coord = self.K @ cam_coord

        u = pixel_coord[0][0] / pixel_coord[2][0]
        v = pixel_coord[1][0] / pixel_coord[2][0]

        # print(f"[debug] world_coord {world_coord} -> pixel_coord {(int(u), int(v))}")

        return (int(u), int(v))

def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()

LABEL_NAME_MAPPING = {
    0:  "Unlabeled",
    1:  "Building",
    2:  "Fence",
    3:  "Other",
    4:  "Pedestrian",
    5:  "Pole",
    6:  "RoadLine",
    7:  "Road",
    8:  "SideWalk",
    9:  "Vegetation",
    10: "Vehicles",
    11: "Wall",
    12: "TrafficSign",
    13: "Sky",
    14: "Ground",
    15: "Bridge",
    16: "RailTrack",
    17: "GuardRail",
    18: "TrafficLight",
    19: "Static",
    20: "Dynamic",
    21: "Water",
    22: "Terrain",
}

ACTUAL_LABEL_NAME_MAPPING = {
    0:  "Void",
    1:  "Road",
    2:  "SideWalk",
    3:  "Building",
    4:  "Wall",
    5:  "Fence",
    6:  "Pole",
    7:  "TrafficLight",
    8:  "TrafficSign",
    9:  "Vegetation",
    10: "Ground",
    11: "Sky",
    12: "Pedestrian",
    13: "Other", # #
    14: "Vehicles",
    15: "Bridge", #
    16: "RailTrack", #
    17: "GuardRail", #
    18: "Unlabeled", # #
    19: "Static", #
    20: "Dynamic", #
    21: "Water", #
    22: "Terrain", #
    24: "RoadLine"
}

    # 3:  "Other",
    # 11: "Wall",
    # 15: "Bridge", #
    # 16: "RailTrack", #
    # 17: "GuardRail", #
    # 19: "Static", #
    # 20: "Dynamic", #
    # 21: "Water", #
    # 22: "Terrain", #

SEMANTIC_COLOR_MAPPING = {
    0:  (0, 0, 0),           # lanemark     # Unlabeled - Unknown or empty space
    1:  (70, 70, 70),        # road         # Building - Buildings and attached structures
    2:  (100, 40, 40),       # sidewalk     # Fence - Fences and barriers
    3:  (55, 90, 80),        # building     # Other - Miscellaneous objects not otherwise classified
    4:  (220, 20, 60),       # wall         # Pedestrian - Pedestrians and non-motorized actors
    5:  (153, 153, 153),     # fence        # Pole - Poles (signs, traffic lights)
    6:  (157, 234, 50),      # pole         # RoadLine - Lane markings and road lines
    7:  (128, 64, 128),      # trafficlight # Road - Road surface
    8:  (244, 35, 232),      # trafficsign  # SideWalk - Sidewalks, bike lanes, traffic islands
    9:  (107, 142, 35),      # vegetation   # Vegetation - Trees, bushes, hedges
    10: (0, 0, 142),         # ground       # Vehicles - All kinds of vehicles
    11: (102, 102, 156),     # sky          # Wall - Standalone walls
    12: (220, 220, 0),       # peds         # TrafficSign - Traffic signs (excluding poles)
    13: (70, 130, 180),      # Sky - Sky, including sun and clouds
    14: (81, 0, 81),         # vehicle      # Ground - Flat ground not part of the road
    15: (150, 100, 100),     # Bridge - Bridges and their structures
    16: (230, 150, 140),     # RailTrack - Train and tram tracks
    17: (180, 165, 180),     # GuardRail - Guardrails and collision barriers
    18: (250, 170, 30),      # TrafficLight - Traffic light heads (excluding poles)
    19: (110, 190, 160),     # Static - Static objects (benches, extinguishers, etc.)
    20: (170, 120, 50),      # Dynamic - Dynamic objects (animals, bins, wheelchairs)
    21: (45, 60, 150),       # Water - Water surfaces (lakes, rivers)
    22: (145, 170, 100),     # Terrain - Natural terrain (grass, sand, soil)
}

LANE_MARK_COLOR = (157, 234, 50)

ACTUAL_SEMANTIC_COLOR_MAPPING = {
    18: (0, 0, 0),           # Unlabeled - Unknown or empty space
    3:  (70, 70, 70),        # Building - Buildings and attached structures
    5:  (100, 40, 40),       # Fence - Fences and barriers
    13: (55, 90, 80),        # Other - Miscellaneous objects not otherwise classified
    12: (220, 20, 60),       # Pedestrian - Pedestrians and non-motorized actors
    6:  (153, 153, 153),     # Pole - Poles (signs, traffic lights)
    24:  LANE_MARK_COLOR,    # RoadLine - Lane markings and road lines
    1:  (128, 64, 128),      # Road - Road surface
    2:  (244, 35, 232),      # SideWalk - Sidewalks, bike lanes, traffic islands
    9:  (107, 142, 35),      # Vegetation - Trees, bushes, hedges
    14: (0, 0, 142),         # Vehicles - All kinds of vehicles
    4:  (102, 102, 156),     # Wall - Standalone walls
    8:  (220, 220, 0),       # TrafficSign - Traffic signs (excluding poles)
    11: (70, 130, 180),      # Sky - Sky, including sun and clouds
    10: (81, 0, 81),         # Ground - Flat ground not part of the road
    15: (150, 100, 100),     # Bridge - Bridges and their structures
    16: (230, 150, 140),     # RailTrack - Train and tram tracks
    17: (180, 165, 180),     # GuardRail - Guardrails and collision barriers
    7:  (250, 170, 30),      # TrafficLight - Traffic light heads (excluding poles)
    19: (110, 190, 160),     # Static - Static objects (benches, extinguishers, etc.)
    20: (170, 120, 50),      # Dynamic - Dynamic objects (animals, bins, wheelchairs)
    21: (45, 60, 150),       # Water - Water surfaces (lakes, rivers)
    22: (145, 170, 100),     # Terrain - Natural terrain (grass, sand, soil)
}

DEFAULT_COLOR = (0, 0, 0)

def decode_semantic_image(label_img):
    height, width = label_img.shape[:2]
    color_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Apply defined colors
    for label_id, color in ACTUAL_SEMANTIC_COLOR_MAPPING.items():
        mask = label_img == label_id
        r, g, b = color
        color_img[mask] = (b, g, r)  # Convert RGB to BGR if needed

    # Identify undefined labels
    unique_labels = np.unique(label_img)
    defined_labels = set(ACTUAL_SEMANTIC_COLOR_MAPPING.keys())
    undefined_labels = [label for label in unique_labels if label not in defined_labels]

    # Assign DEFAULT_COLOR to undefined labels
    for label_id in undefined_labels:
        mask = label_img == label_id
        r, g, b = DEFAULT_COLOR
        color_img[mask] = (b, g, r)  # Convert RGB to BGR if needed
        print(f"BEV Generator Warning: Label ID {label_id} not in ACTUAL_SEMANTIC_COLOR_MAPPING. Assigned DEFAULT_COLOR.")

    return color_img

def process_ego_vehicle(actor, img, intrinsics, extrinsics, layout_projector, history_dict, frame, history_count, step):
    if layout_projector is not None:
        layout_projector._render_vehicles(img, actor)
    draw_point = project_point(actor['location'], intrinsics, extrinsics)
    _, desc, _ = get_vehicle_str(actor, neglect_pos=True)
    if draw_point is not None:
        draw_labeled_dot(img, draw_point[0], draw_point[1], f"the ego vehicle", color=COLOR_BLUE)
    point_list = get_history_points(intrinsics=intrinsics,
                                    extrinsics=extrinsics,
                                    id=actor['id'],
                                    history_dict=history_dict,
                                    frame=frame,
                                    history_count=history_count,
                                    step=step)
    # print(f"[debug] {actor['id']}'s point_list: {point_list}")
    draw_fading_polyline(image=img, points=point_list,
                         color=COLOR_BLUE)
    return img

def process_vehicle(actor, img, intrinsics, extrinsics, layout_projector, history_dict, frame, history_count, step):
    # print(f"[debug] process_vehicle, type(img) = {type(img)}, shape is {img.shape}, type is {img.dtype}")
    if layout_projector is not None:
        layout_projector._render_vehicles(img, actor)
    if actor['state'] == 'dynamic':
        color = COLOR_WHITE
        if 'color' in actor:
            rgb = actor['color'].split(',')
            color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        draw_point = project_point(actor['location'], intrinsics, extrinsics)
        _, desc, _ = get_vehicle_str(actor, neglect_pos=True)
        if draw_point is not None:
            draw_labeled_dot(img, draw_point[0], draw_point[1], f"c{actor['id']}\n{desc}", color=COLOR_WHITE)
        point_list = get_history_points(intrinsics=intrinsics,
                                        extrinsics=extrinsics,
                                        id=actor['id'],
                                        history_dict=history_dict,
                                        frame=frame,
                                        history_count=history_count,
                                        step=step)
        # print(f"[debug] {actor['id']}'s point_list: {point_list}")
        draw_fading_polyline(image=img, points=point_list,
                             color=color)
    else:
        pass
        # ignore static vehicles
        # color = COLOR_GRAY
        # if 'color' in actor:
        #     rgb = actor['color'].split(',')
        #     color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        # draw_point = project_point(actor['location'], intrinsics, extrinsics)
        # _, desc, _ = get_vehicle_str(actor, neglect_pos=True)
        # if draw_point is not None:
        #     draw_labeled_dot(img, draw_point[0], draw_point[1], f"c{actor['id']}\n{desc}", color=COLOR_GRAY)
        # point_list = get_history_points(intrinsics=intrinsics,
        #                                 extrinsics=extrinsics,
        #                                 id=actor['id'],
        #                                 history_dict=history_dict,
        #                                 frame=frame,
        #                                 history_count=history_count,
        #                                 step=step)
        # # print(f"[debug] {actor['id']}'s point_list: {point_list}")
        # draw_fading_polyline(image=img, points=point_list,
        #                      color=color)
    return img

def process_walker(actor, img, intrinsics, extrinsics, layout_projector, history_dict, frame, history_count, step):
    if layout_projector is not None:
        layout_projector._render_walkers(img, actor)
    draw_point = project_point(actor['location'], intrinsics, extrinsics)
    if draw_point is not None:
        draw_labeled_dot(img, draw_point[0], draw_point[1], f"c{actor['id']}\nwalker", color=COLOR_CYAN)
    point_list = get_history_points(intrinsics=intrinsics,
                                    extrinsics=extrinsics,
                                    id=actor['id'],
                                    history_dict=history_dict,
                                    frame=frame,
                                    history_count=history_count,
                                    step=step)
    # print(f"[debug] {actor['id']}'s point_list: {point_list}")
    draw_fading_polyline(image=img, points=point_list,
                         color=COLOR_CYAN)
    return img

def process_traffic_light(actor, img, intrinsics, extrinsics, filter_bottom=False):
    draw_point = project_point(actor['location'], intrinsics, extrinsics)
    state = actor['state']
    # for traffic_light
    # 0 - Red; 1 - Yellow; 2 - Green; 3 - Off; 4 - Unknown;
    state_dict = {
        0: "Red",
        1: "Yellow",
        2: "Green",
        3: "Off",
        4: "Unknown"
    }
    if draw_point is not None and not (filter_bottom and draw_point[1] > img.shape[0] * 0.5):
        draw_traffic_light(img, (draw_point[0], draw_point[1]), state_dict[state])
    return img

def process_traffic_sign(actor, img, intrinsics, extrinsics, filter_bottom=False):
    draw_point = project_point(actor['location'], intrinsics, extrinsics)
    if draw_point is not None and not (filter_bottom and draw_point[1] > img.shape[0] * 0.5):
        draw_labeled_dot(img, draw_point[0], draw_point[1], f"{actor['type_id'].split('.')[-1]}", color=COLOR_YELLOW)
    return img

def process_stop_sign(actor, img, intrinsics, extrinsics, filter_bottom=False):
    draw_point = project_point(actor['location'], intrinsics, extrinsics)
    if draw_point is not None and not (filter_bottom and draw_point[1] > img.shape[0] * 0.5):
        draw_stop_sign(img, [(draw_point[0], draw_point[1])])
    return img

def process_speed_limit(actor, img, intrinsics, extrinsics, filter_bottom=False):
    draw_point = project_point(actor['location'], intrinsics, extrinsics)
    speed_limit_val = actor['type_id'].split('.')[-1]
    if draw_point is not None and not (filter_bottom and draw_point[1] > img.shape[0] * 0.5):
        draw_speed_signs(img, [(draw_point[0], draw_point[1])], [speed_limit_val])
    return img

def get_history_points(intrinsics, extrinsics, history_dict, id, frame, history_count=4, step=5):
    point_list = []
    if id not in history_dict:
        return []
    for i in range (0, history_count + 1):
        index = frame - i * step
        if index not in history_dict[id]:
            return point_list
        pos3d = history_dict[id][index]
        point2d = project_point(pos3d, intrinsics, extrinsics)
        if point2d is None:
            return point_list
        point_list.append((point2d[0], point2d[1]))
    return point_list

def convert_3d_dot_list_to_2d(intrinsics, extrinsics, dot3dlist):
    dot2dlist = []
    for pos3d in dot3dlist:
        point2d = project_point(pos3d, intrinsics, extrinsics)
        if point2d is not None:
            dot2dlist.append(point2d)
    return dot2dlist

def rel_pos_to_str(position, yaw):
    x_str = ""
    y_str = ""
    if position[0] < 0:
        x_str = f"{-position[0]:.1f}m to the rear"
    else:
        x_str = f"{position[0]:.1f}m to the front"
    if position[1] < 0:
        y_str = f"{-position[1]:.1f}m to the left"
    else:
        y_str = f"{position[1]:.1f}m to the right"
    yaw_str = ""

    yaw = (yaw + 180) % 360 - 180

    if yaw > 0:
        yaw_str = f"{abs(yaw):.1f} degrees to the right"
    elif yaw < 0:
        yaw_str = f"{abs(yaw):.1f} degrees to the left"
    
    if abs(yaw) < 1:
        yaw_str = "same direction as the ego vehicle"

    if abs(yaw) > 179:
        yaw_str = "opposite direction to the ego vehicle"
    
    pos_str = f"{x_str}, {y_str}"
    return pos_str, yaw_str


def generate_basic_info_list(anno_data):
    bounding_boxes = anno_data['bounding_boxes']
    ego_vehicle = None
    for actor in bounding_boxes:
        if actor['class'] == 'ego_vehicle':
            ego_vehicle = actor
    
    info_list = []
    if ego_vehicle is not None:
        for actor in bounding_boxes:
            if abs(actor['location'][2] - ego_vehicle['location'][2]) > CONSIDER_Z:
                # avoid the situation that the role actor is initialized undergound
                continue
            if 'distance' in actor and actor['distance'] > ANNO_CONSIDER_DISTANCE:
                continue
            if 'num_points' in actor and actor['num_points'] < 3:
                continue

            actor['position'] = transform_to_ego_coordinates(actor['location'], ego_vehicle['world2ego'])
            actor['yaw'] = actor["rotation"][2] - ego_vehicle["rotation"][2]

            if actor['class'] == 'ego_vehicle':
                info_list.append({
                    "desc": "ego vehicle",
                    "speed": f"{actor['speed']:.1f}m/s"
                })
            elif actor['class'] == 'vehicle':
                if actor['state'] == 'dynamic':
                    _, desc, _ = get_vehicle_str(actor, neglect_pos=True)
                    pos_str, yaw_str = rel_pos_to_str(actor['position'], actor['yaw'])
                    info_list.append({
                        "id": f"c{actor['id']}",
                        "desc": desc,
                        "relative position": pos_str,
                        "speed": f"{actor['speed']:.1f}m/s",
                        "speed direction relative to ego": yaw_str
                    })
            elif actor['class'] == 'walker':
                pos_str, yaw_str = rel_pos_to_str(actor['position'], actor['yaw'])
                info_list.append({
                    "id": f"c{actor['id']}",
                    "desc": "walker",
                    "relative position": pos_str,
                    "speed": f"{actor['speed']:.1f}m/s",
                    "speed direction relative to ego": yaw_str
                })
    return info_list

def generate_anno_bev_image(anno_data, raw_img, carla_map,
                            history_dict, frame, history_count=4, stride=5, 
                            wp_list=None, filter_back_signs=True,
                            filter_affect_ego_signs=True):

    # print(f"[debug] generate_anno_bev_image, type(anno_data) = {type(anno_data)}, type(raw_img) = {type(raw_img)}, type = {raw_img.dtype}, shape = {raw_img.shape}, type(history_dict) = {type(history_dict)}")
    bounding_boxes = anno_data['bounding_boxes']
    bev_sensor_anno = anno_data['sensors']['BEV']

    cam_location = bev_sensor_anno['location']
    cam_rotation = bev_sensor_anno['rotation']
    intrinsic = bev_sensor_anno['intrinsic']
    world2cam = bev_sensor_anno['world2cam']
    cam2ego = bev_sensor_anno['cam2ego']
    fov = bev_sensor_anno['fov']
    img_size_x = bev_sensor_anno['image_size_x']
    img_size_y = bev_sensor_anno['image_size_y']

    ego_vehicle = None
    for actor in bounding_boxes:
        if actor['class'] == 'ego_vehicle':
            ego_vehicle = actor

    img = raw_img

    layout_projector = None
    if MINIMAL <= 0:
        layout_projector = RoadCameraProjector(intrinsic, world2cam,
                                            ego_location=ego_vehicle['location'],
                                            precision=0.05, consider_radius=90.0)
        topology = carla_map.get_topology()
        layout_projector.draw_topology(img=img,
                                    carla_topology=topology, 
                                    index=0)

    if wp_list is not None and not isinstance(wp_list, list):
        wp_list = wp_list.tolist()
    if isinstance(wp_list, list):
        # print("[debug] bev renderer got wp")
        wp2dlist = convert_3d_dot_list_to_2d(intrinsic, world2cam, wp_list)
        # print(f"[debug] wp2dlist = {wp2dlist}")
        draw_dot_list(img, wp2dlist, COLOR_GREEN)

    if ego_vehicle is not None:
        ego_vehicle_list = []
        vehicle_list = []
        walker_list = []
        traffic_light_list = []
        traffic_sign_list = []

        # classify all elements
        for actor in bounding_boxes:
            if abs(actor['location'][2] - ego_vehicle['location'][2]) > CONSIDER_Z:
                # avoid the situation that the role actor is initialized undergound
                continue
            if 'distance' in actor and actor['distance'] > CONSIDER_DISTANCE:
                continue

            if actor['class'] == 'ego_vehicle':
                ego_vehicle_list.append(actor)
            if actor['class'] == 'vehicle':
                vehicle_list.append(actor)
            elif actor['class'] == 'walker':
                walker_list.append(actor)
            elif actor['class'] == 'traffic_light':
                if filter_affect_ego_signs and 'affects_ego' in actor and actor['affects_ego'] == False:
                    continue
                traffic_light_list.append(actor)
            elif actor['class'] == 'traffic_sign' or 'stop' in actor['type_id']:
                if filter_affect_ego_signs and 'affects_ego' in actor and actor['affects_ego'] == False:
                    continue
                traffic_sign_list.append(actor)
        
        # draw all actor layers
        for actor in traffic_sign_list:
            if 'stop' in actor['type_id']:
                img = process_stop_sign(actor=actor, img=img, 
                                        intrinsics=intrinsic, extrinsics=world2cam, 
                                        filter_bottom=filter_back_signs)
            elif 'speed_limit' in actor['type_id']:
                img = process_speed_limit(actor=actor, img=img, 
                                          intrinsics=intrinsic, extrinsics=world2cam,
                                          filter_bottom=filter_back_signs)
            else:
                img = process_traffic_sign(actor=actor, img=img, 
                                           intrinsics=intrinsic, extrinsics=world2cam,
                                           filter_bottom=filter_back_signs)
        for actor in traffic_light_list:
            img = process_traffic_light(actor=actor, img=img, 
                                        intrinsics=intrinsic, extrinsics=world2cam,
                                        filter_bottom=filter_back_signs)
        for actor in vehicle_list:
            img = process_vehicle(actor=actor, img=img, 
                                    intrinsics=intrinsic, extrinsics=world2cam,
                                    layout_projector=layout_projector,
                                    history_dict=history_dict,
                                    frame=frame,
                                    history_count=history_count,
                                    step=stride)
        for actor in walker_list:
            img = process_walker(actor=actor, img=img, 
                                    intrinsics=intrinsic, extrinsics=world2cam,
                                    layout_projector=layout_projector,
                                    history_dict=history_dict,
                                    frame=frame,
                                    history_count=history_count,
                                    step=stride)
        for actor in ego_vehicle_list:
            img = process_ego_vehicle(actor=actor, img=img, 
                                        intrinsics=intrinsic, extrinsics=world2cam,
                                        layout_projector=layout_projector,
                                        history_dict=history_dict,
                                        frame=frame,
                                        history_count=history_count,
                                        step=stride)

    img = cut_to_square(img)
    return img


def generate_anno_rgb_image(anno_data, sensor_key, raw_img, 
                            history_dict, frame, history_count=4, stride=5,
                            filter_back_signs=True):

    # print(f"[debug] generate_anno_rgb_image, type(anno_data) = {type(anno_data)}, type(sensor_key) = {type(sensor_key)}, type(raw_img) = {type(raw_img)}, shape = {raw_img.shape}, type = {raw_img.dtype}, type(history_dict) = {type(history_dict)}")
    # print(f"[debug] history_dict = {history_dict}")
    bounding_boxes = anno_data['bounding_boxes']
    sensor_anno = anno_data['sensors'][sensor_key]

    cam_location = sensor_anno['location']
    cam_rotation = sensor_anno['rotation']
    intrinsic = sensor_anno['intrinsic']
    world2cam = sensor_anno['world2cam']
    cam2ego = sensor_anno['cam2ego']
    fov = sensor_anno['fov']
    img_size_x = sensor_anno['image_size_x']
    img_size_y = sensor_anno['image_size_y']

    ego_vehicle = None
    for actor in bounding_boxes:
        if actor['class'] == 'ego_vehicle':
            ego_vehicle = actor

    img = np.array(raw_img)
    if ego_vehicle is not None:
        for actor in bounding_boxes:
            if abs(actor['location'][2] - ego_vehicle['location'][2]) > CONSIDER_Z:
                # avoid the situation that the role actor is initialized undergound
                continue
            if 'distance' in actor and actor['distance'] > CONSIDER_DISTANCE:
                continue
            if 'num_points' in actor and actor['num_points'] < 3:
                continue

            # if actor['class'] == 'ego_vehicle':
            #     img = process_ego_vehicle(actor=actor, img=img, 
            #                               intrinsics=intrinsic, extrinsics=world2cam,
            #                               history_dict=history_dict,
            #                               frame=frame,
            #                               history_count=history_count,
            #                               step=stride)
            if actor['class'] == 'vehicle':
                img = process_vehicle(actor=actor, img=img, 
                                      intrinsics=intrinsic, extrinsics=world2cam,
                                      layout_projector=None,
                                      history_dict=history_dict,
                                      frame=frame,
                                      history_count=history_count,
                                      step=stride)
            elif actor['class'] == 'walker':
                img = process_walker(actor=actor, img=img, 
                                     intrinsics=intrinsic, extrinsics=world2cam,
                                     layout_projector=None,
                                     history_dict=history_dict,
                                     frame=frame,
                                     history_count=history_count,
                                     step=stride)

    return img

def add_top_text(image, text, font_scale=2, thickness=3, color=(255, 255, 255), margin=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + margin
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def main():
    # Create a blank image (for demo)
    height, width = 480, 640
    color_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Draw traffic lights at various positions
    draw_traffic_light(color_img, (100, 100), "Red")
    draw_traffic_light(color_img, (200, 100), "Yellow")
    draw_traffic_light(color_img, (300, 100), "Green")
    draw_traffic_light(color_img, (400, 100), "Off")
    draw_traffic_light(color_img, (500, 100), "Unknown")
    draw_traffic_light(color_img, (320, 300), "Red", size=80)  # Larger one

    centers = [(100, 100), (300, 200), (500, 300)]
    limits = ["30", "50", "80"]

    stop_centers = [(150, 100), (400, 250), (650, 120)]
    draw_stop_sign(color_img, stop_centers)

    draw_speed_signs(color_img, centers, limits)

    draw_labeled_dot(color_img, 100, 100, "A", color=(255, 0, 0))
    draw_labeled_dot(color_img, 200, 150, "B", color=(0, 255, 0), font_scale=0.8, radius=10)
    draw_labeled_dot(color_img, 300, 250, "C", color=(0, 0, 255), font_scale=1.0, radius=12)

    pts = [(100, 100), (200, 150), (250, 250), (300, 200), (400, 300)]
    draw_fading_polyline(color_img, pts, color=COLOR_BLUE, thickness=4)

    # Save the result
    output_path = "demo.png"
    cv2.imwrite(output_path, color_img)
    print(f"Demo image saved to {output_path}")

if __name__ == "__main__":
    main()