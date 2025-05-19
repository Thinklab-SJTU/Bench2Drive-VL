from webcolors import (
    CSS2_HEX_TO_NAMES,
    hex_to_rgb,
)
from scipy.spatial import KDTree

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
        if 10 < r < 60 and 10 < g < 20 and 10 < b < 20:
            final_name = 'dark red'
        if 10 < r < 20 and 10 < g < 60 and 10 < b < 20:
            final_name = 'dark green'
        if 10 < r < 20 and 10 < g < 20 and 10 < b < 60:
            final_name = 'dark blue'
    return f'{final_name}'


print(convert_rgb_to_names((0, 12, 58)))