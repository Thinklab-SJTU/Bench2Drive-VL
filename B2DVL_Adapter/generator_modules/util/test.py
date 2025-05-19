from waypoint_encoder import *
from waypoint_decoder import *
from waypoint_extractor import *

sequence = [[0, 0], [0, 3], [0, -3], [1, 0], [-1, 0], [2.33, 2.33]]
xy_code = generate_motion_tokens(sequence)
ds_code = generate_motion_and_direction_tokens(sequence)
xy_decd = decode_xy_token(''.join(xy_code))
ds_decd = decode_polar_token(''.join(ds_code))

print(sequence)
print(xy_code)
print(ds_code)
print(xy_decd)
print(ds_decd)

raw_str = "{\"0.5s\": [0.04, 0.0], \"1.0s\": [1.09, -0.0], \"1.5s\": [3.65, -0.0], \"2.0s\": [7.7, -0.01], \"2.5s\": [12.15, -0.07], \"3.0s\": [16.11, -0.12], \"3.5s\": [20.42, -0.16], \"4.0s\": [24.62, -0.18]}"

wp_json = json.loads(raw_str)

deltas, xy_code, ds_code = extract_delta_and_token_from_json(wp_json)
xy_decd = decode_xy_token(''.join(xy_code))
ds_decd = decode_polar_token(''.join(ds_code))

print("\n\n\n\n\n")
print(raw_str)
print(deltas)
print(xy_code)
print(ds_code)
print(xy_decd)
print(ds_decd)
print(get_waypoint_dict_seq_from_rel(xy_decd))
print(get_waypoint_dict_seq_from_rel(ds_decd))
