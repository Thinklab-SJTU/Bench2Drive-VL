import re
import math

def hex_to_float(hex_str):
    return int(hex_str, 16) / 16.0

def hex_to_anglef(hex_str):
    return int(hex_str, 16) * 180.0 / 64.0

def decode_xy_token(encoded_str):
    pattern = re.findall(r"<(x|y)_(stay|pos|neg|fwd|back|right|left)_([0-9a-fA-F]{2})>", encoded_str)
    
    waypoints = []
    
    x, y = 0.0, 0.0
    for axis, direction, hex_val in pattern:
        move_val = hex_to_float(hex_val)

        if axis == 'x':
            if direction == "stay":
                move_val = 0.0
            elif direction == "pos" or direction == "fwd":
                x = move_val
            elif direction == "neg" or direction == "back":
                x = -move_val

        elif axis == 'y':
            if direction == "stay":
                move_val = 0.0
            elif direction == "pos" or direction == "right":
                y = move_val
            elif direction == "neg" or direction == "left":
                y = -move_val
            waypoints.append([x, y])
            x, y = 0.0, 0.0

    return waypoints

def decode_polar_token(encoded_str):
    pattern = re.findall(r"<(dir)_(fwd|back|right|left)_([0-9a-fA-F]{2})>|<(spd)_([0-9a-fA-F]{2})>", encoded_str)
    
    waypoints = []
    x, y = 0.0, 0.0
    angle = 0

    for dir_tag, direction, hex_val_dir, spd_tag, hex_val_spd in pattern:
        if dir_tag == "dir":
            value = hex_to_anglef(hex_val_dir)
            if hex_val_dir == "00":
                angle = 0
            elif hex_val_dir == "40":
                angle = 180
            else:
                angle = value
                if direction == "left":
                    angle = -angle
                elif direction == "right":
                    angle = angle
                elif direction == "fwd":
                    angle = 0
                elif direction == "back":
                    angle = 180

        elif spd_tag == "spd":
            distance = hex_to_float(hex_val_spd)
            x = round(distance * math.cos(math.radians(angle)), 4)
            y = round(distance * math.sin(math.radians(angle)), 4)
            waypoints.append([x, y])
            x, y = 0.0, 0.0
            angle = 0

    return waypoints
